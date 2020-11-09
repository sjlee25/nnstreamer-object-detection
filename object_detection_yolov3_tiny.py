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
    b) $ python3 object_detection_tf.py [od_model] [od_label]

See https://lazka.github.io/pgi-docs/#Gst-1.0 for Gst API details.
"""

from argparse import ArgumentParser
import os
import sys
import logging
import gi
from math import exp
import cairo
import struct
import numpy as np
import colorsys
import time
import pandas as pd
from datetime import datetime
import cv2 # for temporarily get video size

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GstVideo

import gt_position_extractor
import label_mapper

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
        parser = ArgumentParser()
        parser.add_argument('--model', type=str, help='tf model path')
        parser.add_argument('--label', type=str, help='label path')
        parser.add_argument('--use_web_cam', action='store_true', help='to use web cam')
        parser.add_argument('--file', type=str, help='file path')
        parser.add_argument('--train_folder', type=str, help='for train set')
        parser.add_argument('--threshold_score', type=float,)

        args = parser.parse_args()

        self.od_framework= 'tensorflow'
        self.od_model = args.model if args.model else './models/yolo_v3/frozen_yolov3_tiny.pb'
        self.od_label = args.label if args.label else './models/yolo_v3/coco.names'
        self.use_web_cam = args.use_web_cam if args.use_web_cam else False
        self.file_path = args.file if args.file else './video/test_video_street.mp4'
        self.train_folder_path = args.train_folder if args.train_folder else 'train'

        self.loop = None
        self.pipeline = None
        self.running = False
        
        self.labels = {}
        self.bboxes = []
        self.times = []

        vid = cv2.VideoCapture(self.file_path)
        self.video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.model_width = 416
        self.model_height = 416

        self.resize_ratio = min(self.model_width / self.video_width, self.model_height / self.video_height)
        self.dw = (self.model_width - self.resize_ratio * self.video_width) / 2.
        self.dh = (self.model_height - self.resize_ratio * self.video_height) / 2.

        self.box_size = 4
        self.num_labels = 80

        # threshold values to drop detections
        self.iou_threshold = 0.5
        self.score_threshold = 0.3
        # self.score_threshold = args.score_threshold if args.score_threshold else self.score_threshold

        # cairo overlay state
        self.cairo_valid = True
        self.valid_scale=[0, np.inf]

        # load labels
        with open(self.od_label, 'r') as data:
            for ID, name in enumerate(data):
                self.labels[ID] = name.strip('\n')

        self.mapper = label_mapper.LabelMapper(self.od_label)
        self.labels = self.mapper.labels

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

        # gstreamer pipeline
        if self.use_web_cam :
            pipeline = 'v4l2src name=cam_src ! videoscale '
        else :
            pipeline = f'filesrc location={self.file_path} ! decodebin name=decode decode. ! videoscale ! videorate '

        self.pipeline = Gst.parse_launch(pipeline + 
            '! videoconvert ! video/x-raw,width=%d,height=%d,format=RGB ! tee name=t ' % (self.video_width, self.video_height) +
            't. ! queue ! videoconvert ! timeoverlay ! textoverlay name=text_overlay ! cairooverlay name=tensor_res tensor_res. ! fpsdisplaysink name=fps_sink video-sink=ximagesink text-overlay=false signal-fps-measurements=true '
            't. ! queue leaky=2 max-size-buffers=1 ! videoscale add-borders=1 ! video/x-raw,width=%d,height=%d,format=RGB,framerate=24/1,pixel-aspect-ratio=1/1 ! ' % (self.model_width, self.model_height) + 
                'tensor_converter input-dim=3:%d:%d:1 ! tensor_transform mode=typecast option=float32 ! ' % (self.model_width, self.model_height) +
                'tensor_filter framework=%s model=%s ' % (self.od_framework, self.od_model) + 
                    'input=3:%d:%d:1 inputname=inputs inputtype=float32 ' % (self.model_width, self.model_height) + 
                    'output=85:2535:1 outputname=output_boxes outputtype=float32 ! '
                'tensor_sink name=tensor_sink'
        )
        print('[Info] Pipeline checked!')

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

        # frame count
        self.frame_by_bin = self.pipeline.get_by_name('decode')
        self.frame = 0

        # set window title
        self.set_window_title('img_tensor', 'YOLOv3-tiny OD')

        # mesuare fps
        fps_sink = self.pipeline.get_by_name('fps_sink')
        fps_sink.connect('fps-measurements', self.on_fps_message)

        # make out file
        folder_path = './output/'
        if('train' in self.file_path):
            folder_path = folder_path + self.train_folder_path + '/' + self.file_path.split('/')[-1].split('.')[0]
        else:
            folder_path = folder_path + self.file_path.split('/')[-1].split('.')[0]
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        detected_folder_path = folder_path + '/detections'
        if not os.path.exists(detected_folder_path):
            os.makedirs(detected_folder_path)

        fps_folder_path = folder_path + '/fps'
        if not os.path.exists(fps_folder_path):
            os.makedirs(fps_folder_path)
                
        detected_objects_csv_path = detected_folder_path + f'/all_detections.csv'

        i = 0
        while True:
            if not os.path.isfile(detected_objects_csv_path):
                break
            else:
                i += 1
                if i == 1:
                    detected_objects_csv_path = detected_objects_csv_path.replace('.csv', '_1.csv')
                    if not os.path.isfile(detected_objects_csv_path):
                        break
                else:
                    if f'_{i-1}.csv' in detected_objects_csv_path:
                        detected_objects_csv_path = detected_objects_csv_path.replace(f'_{i-1}.csv', f'_{i}.csv')
                    else:
                        detected_objects_csv_path = detected_objects_csv_path.replace('.csv', f'_{i}.csv')

        fps_file_path = fps_folder_path + f'/{self.score_threshold}.csv'

        i = 0
        while True:
            if not os.path.isfile(fps_file_path):
                break
            else:
                i += 1
                if i == 1:
                    fps_file_path = fps_file_path.replace('.csv', '_1.csv')
                    if not os.path.isfile(fps_file_path):
                        break
                else:
                    if f'_{i-1}.csv' in fps_file_path:
                        fps_file_path = fps_file_path.replace(f'_{i-1}.csv', f'_{i}.csv')
                    else:
                        fps_file_path = fps_file_path.replace('.csv', f'_{i}.csv')

        self.detected_objects_data = []
        self.fps_data = []
       
        self.gt_objects = {}
        # self.draw_gt_box = True
        self.draw_gt_box = False
        if self.draw_gt_box:
            file_path = self.file_path.split('/')[-1]
            extension_idx = file_path.rfind('.')
            file_path = file_path[:extension_idx]
            self.gt_objects = gt_position_extractor.GtPositionExtractor(file_path).get_gtobjects_from_csv()
            print(f'''[Info] Read ground truth boxes in {len(self.gt_objects)} frames''')

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        # run main loop
        self.loop.run()

        # quit when received eos or error message
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()

        # write output
        self.csv = pd.DataFrame(self.fps_data, columns = ['fps', 'tensor_fps'])
        self.csv.to_csv(fps_file_path, index=False, mode = 'w')
        self.csv = pd.DataFrame(self.detected_objects_data)
        self.csv.to_csv(detected_objects_csv_path, index=False, header=False, mode='w')

        # calculate average
        interval = (self.times[-1] - self.times[0])
        print(f"average tensor fps: {(len(self.times)-1)/interval}")

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
        success, self.frame = self.frame_by_bin.query_position(Gst.Format.DEFAULT)
        
        bboxes, bboxes_all = self.postprocess_boxes(pred_bbox)
        bboxes_all = self.nms(bboxes_all, method='nms')
        for bbox in bboxes_all:
            self.detected_objects_data.append([str(self.frame).zfill(6), bbox[0], bbox[1], bbox[2], bbox[3], self.mapper.get_data_set_label(int(bbox[5])), bbox[4]])

        self.bboxes = self.nms(bboxes, method='nms')
        self.times.append(time.time())
        
    def draw_overlay_cb(self, overlay, context, timestamp, duration):      
        if not self.cairo_valid or not self.running:
            return
    
        # draw_cnt = 0
        context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(20)
        context.set_line_width(3.0)
        context.set_source_rgb(0.1, 1.0, 0.8)

        # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        for detected_object in self.bboxes:
            x = int(detected_object[0])
            y = int(detected_object[1])
            width = int(detected_object[2]) - x
            height = int(detected_object[3]) - y

            score = detected_object[4]
            label_idx = int(detected_object[5])
            label = self.labels[label_idx]

            # box_color = self.colors[label_idx]
            # context.set_source_rgb(*box_color)
           
            # draw rectangle
            context.rectangle(x, y, width, height)
            context.stroke()

            # draw title
            context.move_to(x - 1, y - 8)
            context.show_text('%s  %.2f' % (label, score))

            # draw_cnt += 1
            # if draw_cnt >= self.max_object_detection:
            #     break

        if not self.draw_gt_box:
            return

        context.set_line_width(2.0)
        context.set_source_rgb(1.0, 1.0, 0.)

        for gt_object in self.gt_objects[str(self.frame)]:
            x = gt_object.x
            y = gt_object.y
            width = gt_object.width
            height = gt_object.height
            # label = gt_object.class_name

            # draw rectangle
            context.rectangle(x, y, width, height)
            context.stroke()

            # draw title
            # context.move_to(x - 1, y - 8)
            # context.show_text('%s  %.2f' % (label))

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
        score_mask_all = scores > 0.

        mask_all = np.logical_and(scale_mask, score_mask_all)
        mask = np.logical_and(scale_mask, score_mask)

        coors_all, scores_all, classes_all = pred_coor[mask_all], scores[mask_all], classes[mask_all]
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        decoded_bbox = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
        decoded_bbox_all = np.concatenate([coors_all, scores_all[:, np.newaxis], classes_all[:, np.newaxis]], axis=-1)

        return decoded_bbox, decoded_bbox_all

    def on_fps_message(self, fpsdisplaysink, fps, droprate, avgfps):
        if len(self.times) >= 2:
            interval = self.times[-1] - self.times[-2]
            label = 'video-fps: %.2f  overlay-fps: %.2f' % (fps, 1/interval)
            textoverlay = self.pipeline.get_by_name('text_overlay')
            textoverlay.set_property('text', label)
            self.fps_data.append([fps, 1/interval])

if __name__ == '__main__':
    od_instance = ObjectDetection()
    od_instance.run()
