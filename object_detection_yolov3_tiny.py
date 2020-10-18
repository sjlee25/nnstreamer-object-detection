# !/usr/bin/env python

"""
NNStreamer example for image classification using tensorflow.

Pipeline :
v4l2src -- tee -- textoverlay -- videoconvert -- ximagesink
            |
            --- videoscale -- tensor_converter -- tensor_filter -- tensor_sink

This app displays video sink.

'tensor_filter' for image classification.
Get model by
$ cd $NNST_ROOT/bin
$ bash get-model.sh object-detection-tf

'tensor_sink' updates classification result to display in textoverlay.

Run example :
Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plugin.
$ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
    a) $ python3 object_detection_tf.py
    b) $ python3 object_detection_tf.py [model_path] [label_path]

See https://lazka.github.io/pgi-docs/#Gst-1.0 for Gst API details.
"""

import os
import sys
import logging
import gi
from math import exp
import cairo
import struct
import numpy as np
import colorsys

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GstVideo

class DetectedObject:
    def __init__(self, x, y, width, height, class_id, score):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_id = class_id
        self.score = score

class ObjectDetection:
    """Object Detection with NNStreamer."""

    def __init__(self, argv=None):
        if len(sys.argv) != 1 and len(sys.argv) != 3:
            print('usage: python3 object_detection_tf.py [model path] [label path]')
            exit(1)

        self.od_framework= 'tensorflow'
        if len(sys.argv) == 1:
            self.model_path = './models/yolo_v3/frozen_yolov3_tiny.pb'
            self.label_path = './models/yolo_v3/coco.names'
        else:
            self.model_path = sys.argv[1]
            self.label_path = sys.argv[2]
        self.file_path = 'video/test_video_street.mp4'

        self.loop = None
        self.pipeline = None
        self.running = False
        
        self.labels = {}
        self.bboxes = []

        self.video_width = 1280
        self.video_height = 720
        self.model_width = 416
        self.model_height = 416

        self.resize_ratio = min(self.model_width / self.video_width, self.model_height / self.video_height)
        self.dw = (self.model_width - self.resize_ratio * self.video_width) / 2.
        self.dh = (self.model_height - self.resize_ratio * self.video_height) / 2.
        print(self.resize_ratio, self.dw, self.dh)

        self.box_size = 4
        self.num_labels = 80

        # threshold values to drop detections
        self.iou_threshold = 0.45
        self.score_threshold = 0.3

        # cairo overlay state
        self.cairo_valid = True
        self.valid_scale=[0, np.inf]

        # load labels
        with open(self.label_path, 'r') as data:
            for ID, name in enumerate(data):
                self.labels[ID] = name.strip('\n')

        # set colors for overlay
        hsv_tuples = [(1.0 * x / self.num_labels, 1., 1.) for x in range(self.num_labels)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        GObject.threads_init()
        Gst.init(argv)

    def run(self):
        """Init pipeline and run example.

        :return: None
        """
        # main loop
        self.loop = GObject.MainLoop()

        # new tf pipeline (NHWC format)
        self.pipeline = Gst.parse_launch(
            # 'v4l2src name=cam_src ! videoscale ! videoconvert ! video/x-raw,width=%d,height=%d,format=RGB,framerate=30/1 ! tee name=t ' % (self.video_width, self.video_height) +
            'filesrc location=%s ! decodebin ! videoscale ! videorate ! videoconvert !  '
            'video/x-raw,width=%d,height=%d,format=RGB,framerate=30/1 ! tee name=t ' % (self.file_path, self.video_width, self.video_height) +
            't. ! queue ! videoconvert ! timeoverlay ! cairooverlay name=tensor_res ! ximagesink name=img_tensor '
            't. ! queue leaky=2 max-size-buffers=1 ! videoscale add-borders=1 ! video/x-raw,width=%d,height=%d,format=RGB,framerate=30/1,pixel-aspect-ratio=1/1 ! ' % (self.model_width, self.model_height) + 
                'tensor_converter input-dim=3:%d:%d:1 ! tensor_transform mode=typecast option=float32 ! ' % (self.model_width, self.model_height) +
                'tensor_filter framework=%s model=%s ' % (self.od_framework, self.model_path) + 
                    'input=3:%d:%d:1 inputname=inputs inputtype=float32 ' % (self.model_width, self.model_height) + 
                    'output=85:2535:1 outputname=output_boxes outputtype=float32 ! '
                'tensor_sink name=tensor_sink'
        )
        print('Pipeline checked!')

        # bus and message callback
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

        # tensor sink signal : new data callback
        tensor_sink = self.pipeline.get_by_name('tensor_sink')
        tensor_sink.connect('new-data', self.on_new_data)

        # cairo overlay : boxes for detected objects
        overlay = self.pipeline.get_by_name('tensor_res')
        overlay.connect('draw', self.draw_overlay_cb)
        # overlay.connect('caps-changed', self.prepare_overlay_cb)

        # set window title
        self.set_window_title('img_tensor', 'YOLOv3-tiny OD')

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        # run main loop
        self.loop.run()

        # quit when received eos or error message
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()

    def on_bus_message(self, bus, message):
        """Callback for message.

        :param bus: pipeline bus
        :param message: message from pipeline
        :return: None
        """
        if message.type == Gst.MessageType.EOS:
            logging.info('received eos message')
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logging.warning('[error] %s : %s', error.message, debug)
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            error, debug = message.parse_warning()
            logging.warning('[warning] %s : %s', error.message, debug)
        elif message.type == Gst.MessageType.STREAM_START:
            logging.info('received start message')
        elif message.type == Gst.MessageType.QOS:
            data_format, processed, dropped = message.parse_qos_stats()
            format_str = Gst.Format.get_name(data_format)
            logging.debug('[qos] format[%s] processed[%d] dropped[%d]', format_str, processed, dropped)

    def get_arr_from_buffer(self, buffer, idx, expected_size, data_type):
        buffer_content = buffer.get_memory(idx)
        result, mapinfo_content = buffer_content.map(Gst.MapFlags.READ)

        if result:
            if mapinfo_content.size == expected_size*4:
                content_arr = np.array(struct.unpack(str(expected_size)+data_type, mapinfo_content.data), dtype=np.float32)
                buffer_content.unmap(mapinfo_content)
                return content_arr
        else:
            buffer_content.unmap(mapinfo_content)
            print('Error: getting memory from buffer with index %d failed' % (idx))
            exit(1)

    def on_new_data(self, sink, buffer):
        """Callback for tensor sink signal.

        :param sink: tensor sink element
        :param buffer: buffer from element
        :return: None
        """

        if not self.running or buffer.n_memory() != 1:
            return
        
        # output shape: [1][2535 = 3 * (13*13 + 26*26)][85 = 4(x_min, y_min, x_max, y_max) + 1(confidence) + 80(class scores)]
        pred_bbox = self.get_arr_from_buffer(buffer, 0, 2535*85, 'f').reshape((-1, 85))
        
        bboxes = self.postprocess_boxes(pred_bbox)
        self.bboxes = self.nms(bboxes, method='nms')
        
    def draw_overlay_cb(self, overlay, context, timestamp, duration):      
        if not self.cairo_valid or not self.running:
            return
    
        # draw_cnt = 0
        context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(20)
        context.set_line_width(2.5)

        # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        for detected_object in self.bboxes:
            x = int(detected_object[0])
            y = int(detected_object[1])
            width = int(detected_object[2]) - x
            height = int(detected_object[3]) - y

            score = detected_object[4]
            label_idx = int(detected_object[5])
            label = self.labels[label_idx]

            box_color = self.colors[label_idx]
            context.set_source_rgb(*box_color)
           
            # draw rectangle
            context.rectangle(x, y, width, height)
            context.stroke_preserve()

            # draw title
            context.move_to(x - 1, y - 8)
            context.show_text('%s  %.2f' % (label, score))

            # draw_cnt += 1
            # if draw_cnt >= self.max_object_detection:
            #     break

    # def prepare_overlay_cb(self, overlay, caps):
    #     # print(self, overlay, caps)
    #     # self.cairo_valid = GstVideo.VideoInfo.from_caps(caps)
    #     self.cairo_valid = True

    def set_window_title(self, name, title):
        """Set window title.

        :param name: GstXImageSink element name
        :param title: window title
        :return: None
        """
        element = self.pipeline.get_by_name(name)
        if element is not None:
            pad = element.get_static_pad('sink')
            if pad is not None:
                tags = Gst.TagList.new_empty()
                tags.add_value(Gst.TagMergeMode.APPEND, 'title', title)
                pad.send_event(Gst.Event.new_tag(tags))

    def bboxes_iou(self, boxes1, boxes2):
        # boxes1 = np.array(boxes1)
        # boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    def nms(self, bboxes, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = self.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > self.iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    def postprocess_boxes(self, pred_bbox):
       #  pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        # pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5, pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        pred_coor = pred_xywh

        # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        pred_coor[:, 0::2] = (pred_coor[:, 0::2] - self.dw) / self.resize_ratio
        pred_coor[:, 1::2] = (pred_coor[:, 1::2] - self.dh) / self.resize_ratio

        # # (3) clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [self.video_width - 1, self.video_height - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0
        pred_coor = np.array(pred_coor, dtype=np.uint16)

        # # (4) discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((self.valid_scale[0] < bboxes_scale), (bboxes_scale < self.valid_scale[1]))

        # # (5) discard some boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores >= self.score_threshold
        
        # apply masks
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        decoded_bbox = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        return decoded_bbox

if __name__ == '__main__':
    od_instance = ObjectDetection()
    od_instance.run()
