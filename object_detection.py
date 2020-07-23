# !/usr/bin/env python

"""
NNStreamer example for image classification using tensorflow-lite.

Pipeline :
v4l2src -- tee -- textoverlay -- videoconvert -- ximagesink
            |
            --- videoscale -- tensor_converter -- tensor_filter -- tensor_sink

This app displays video sink.

'tensor_filter' for image classification.
Get model by
$ cd $NNST_ROOT/bin
$ bash get-model.sh image-classification-tflite

'tensor_sink' updates classification result to display in textoverlay.

Run example :
Before running this example, GST_PLUGIN_PATH should be updated for nnstreamer plugin.
$ export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:<nnstreamer plugin path>
$ python nnstreamer_example_image_classification_tflite.py

See https://lazka.github.io/pgi-docs/#Gst-1.0 for Gst API details.
"""

import os
import sys
import logging
import gi
from math import exp
import cairo
import struct

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GstVideo


class DetectedObject:
    def __init__(self, x, y, width, height, class_id, prob):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.class_id = class_id
        self.prob = prob

class ObjectDetection:
    """Object Detection with NNStreamer."""

    def __init__(self, argv=None):
        self.loop = None
        self.pipeline = None
        self.running = False
        self.current_label_index = -1
        self.new_label_index = -1
        self.new_label_score = 0
        
        self.tflite_model = ''
        self.label_path = ''
        self.box_path = ''

        self.labels = []
        self.boxes = []
        self.detected_objects = []

        self.x_scale = 10.0
        self.y_scale = 10.0
        self.w_scale = 5.0
        self.h_scale = 5.0

        self.video_width = 640
        self.video_height = 480
        self.model_width = 300
        self.model_height = 300

        self.box_size = 4
        self.label_size = 91
        self.detection_max = 1917
        
        # max objects in display
        self.max_object_detection = 5

        # cairo overlay state
        # struct CairoOverlayState in the c++ code
        self.cairo_valid = None
        self.cairo_vinfo = None

        if not self.tflite_init():
            raise Exception

        GObject.threads_init()
        Gst.init(argv)

    def run(self):
        """Init pipeline and run example.

        :return: None
        """
        # main loop
        self.loop = GObject.MainLoop()

        # init pipeline
        self.pipeline = Gst.parse_launch(
            'v4l2src name=cam_src ! videoconvert ! videoscale ! video/x-raw,width=%d,height=%d,format=RGB,framerate=30/1 ! tee name=t ' % (self.video_width, self.video_height) +
            't. ! queue ! videoconvert ! cairooverlay name=tensor_res ! ximagesink name=img_tensor '
            't. ! queue leaky=2 max-size-buffers=2 ! videoscale ! video/x-raw,width=%d,height=%d,format=RGB ! tensor_converter ! ' % (self.model_width, self.model_height) +
            'tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! '
            'tensor_filter framework=tensorflow-lite model=%s ! ' % (self.tflite_model) + 
            'tensor_sink name=tensor_sink'
        )

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
        overlay.connect('caps-changed', self.prepare_overlay_cb)

        # timer to update result
        GObject.timeout_add(200, self.on_timer_update_result)

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        # set window title
        self.set_window_title('img_tensor', 'Object Detection with NNStreamer')

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

    # def compare_objs(self, a, b):
    #     """Compare score of detected objects.

    #     :param a, b: instances of the class DetectedObject
    #     :return: True if the probability of a is higher
    #     """
    #     return a.prob > b.prob

    def iou(self, a, b):
        """Intersection of union.

        :param a, b: instances of the class DetectedObject
        :return: ratio of intersection area if it is positive or else 0
        """
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.width, b.x + b.width)
        y2 = min(a.y + a.height, b.y + b.height)
        w = max(0, (x2 - x1 + 1))
        h = max(0, (y2 - y1 + 1))
        inter = w * h
        area_a = a.width * a.height
        area_b = b.width * b.height
        ratio = inter / (area_a + area_b - inter)

        return ratio >= 0

    # pass only one box which has higher probability out of two
    # if their iou value is higher than the threshold
    def nms(self, detected_objs):
        threshold_iou = 0.5
        num_boxes = len(detected_objs)
        detected_objs = sorted(detected_objs, key=lambda obj: obj.prob, reverse=True)
        to_delete = [False] * num_boxes

        for i in range(num_boxes):
            for j in range(i + 1, num_boxes):
                if self.iou(detected_objs[i], detected_objs[j]) > threshold_iou:
                    to_delete[j] = True

        self.detected_objects.clear()
        for i in range(num_boxes):
            if not to_delete[i]:
                self.detected_objects.append(detected_objs[i])

    def get_detected_objects(self, detections, boxes):
        threshold_score = 0.5
        detected = []

        print('len: ', len(detections))

        for i in range(self.detection_max):
            y_center = boxes[0] / self.y_scale * self.boxes[2][i] + self.boxes[0][i]
            x_center = boxes[1] / self.x_scale * self.boxes[3][i] + self.boxes[1][i]
            h = exp(boxes[2] / self.h_scale) * self.boxes[2][i]
            w = exp(boxes[3] / self.w_scale) * self.boxes[3][i]

            y_min = y_center - h/2.
            x_min = x_center - w/2.
            y_max = y_center + h/2.
            x_max = x_center + w/2.

            x = x_min * self.model_width
            y = y_min * self.model_height
            width = (x_max - x_min) * self.model_width
            height = (y_max - y_min) * self.model_height

            for c in range(1, self.label_size):
                score = 1. / (1. + exp(-detections[c]))
                if score < threshold_score:
                    continue
                print(score)

                detected_obj = DetectedObject(x, y, w, h, c, score)
                detected.append(detected_obj)
        
            # check original code
            # detections += self.label_size
            # boxes += self.box_size

        print(detected)
        self.nms(detected)

    def on_new_data(self, sink, buffer):
        """Callback for tensor sink signal.

        :param sink: tensor sink element
        :param buffer: buffer from element
        :return: None
        """
        if self.running and buffer.n_memory() == 2:
            mem_boxes = buffer.get_memory(0) # check if correct
            result, mapinfo_boxes = mem_boxes.map(Gst.MapFlags.READ)
            if result:
                if mapinfo_boxes.size == self.box_size * self.detection_max * 4:
                    boxes = struct.unpack(str(len(mapinfo_boxes.data)//4) + 'f', mapinfo_boxes.data)
                else: print('failed')
                    
            mem_detections = buffer.get_memory(1) # check if correct
            result, mapinfo_detections = mem_detections.map(Gst.MapFlags.READ)
            if result:
                if mapinfo_detections.size == self.label_size * self.detection_max * 4:
                    detections = struct.unpack(str(len(mapinfo_detections.data)//4) + 'f', mapinfo_detections.data)
                    self.get_detected_objects(detections, boxes)
                    mem_boxes.unmap(mapinfo_boxes)
                    mem_detections.unmap(mapinfo_detections)

    # needed?
    def prepare_overlay_cb(self, overlay, caps):
        # print(self, overlay, caps)
        # self.cairo_valid = GstVideo.VideoInfo.from_caps(caps)
        # print('prepare_overlay_cb: cairo_valid = %d' % (self.cairo_valid))
        self.cairo_valid = True

    def draw_overlay_cb(self, overlay, context, timestamp, duration):      
        if not self.cairo_valid or not self.running:
            return
        
        draw_cnt = 0
        detected = self.detected_objects

        context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(20)

        for detected_object in detected:
            label = self.labels[detected_object.class_id]

            x = detected_object.x * self.video_width / self.model_width
            y = detected_object.y * self.video_height / self.model_height
            width = detected_object * self.video_width / self.model_width
            height = detected_object * self.video_height / self.model_height

            # draw rectangle
            context.rectangle(x, y, width, height)
            context.set_source_rgb(1, 0, 0)
            context.set_line_width(1.5)
            context.stroke()
            context.fill_preserve()

            # draw title
            context.move_to(x + 5, y + 25)
            context.text_path(label)
            context.set_source_rgb(1, 0, 0)
            context.fill_reserve()
            context.set_source_rb(1, 1, 1)
            context.set_line_width(0.3)
            context.stroke()
            context.fill_preserve()

            draw_cnt += 1
            if draw_cnt >= self.max_object_detection:
                break

    def on_timer_update_result(self):
        """Timer callback for textoverlay.

        :return: True to ensure the timer continues
        """
        if self.running:
            if self.current_label_index != self.new_label_index:
                # update textoverlay
                self.current_label_index = self.new_label_index
                label = self.tflite_get_label(self.current_label_index)
                textoverlay = self.pipeline.get_by_name('tensor_res')
                textoverlay.set_property('text', '%s (%.1f)'%(label, self.score))
        return True

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

    def tflite_init(self):
        """Check tflite model and load labels.

        :return: True if successfully initialized
        """
        tflite_model = 'ssd_mobilenet_v2_coco.tflite'
        tflite_label = 'labels.txt'
        tflite_box = 'box_priors.txt'
        current_folder = os.path.dirname(os.path.abspath(__file__))
        model_folder = os.path.join(current_folder, 'tflite_model')

        # check model file exists
        self.tflite_model = os.path.join(model_folder, tflite_model)
        if not os.path.exists(self.tflite_model):
            logging.error('cannot find tflite model [%s]', self.tflite_model)
            return False

        # load box priors
        self.box_path = os.path.join(model_folder, tflite_box)
        try:
            with open(self.box_path, 'r') as box_file:
                for line in box_file.readlines():
                    box_line = line.split(' ')
                    del box_line[0:8]
                    box_line = box_line[:-1]
                    box_line = list(map(float, box_line))
                    if len(box_line) > 0:
                        self.boxes.append(box_line)
        except FileNotFoundError:
            logging.error('cannot find tflite box priors [%s]', self.box_path)
            return False
        logging.info('finished to load box priors')

        # load labels
        self.label_path = os.path.join(model_folder, tflite_label)
        try:
            with open(self.label_path, 'r') as label_file:
                for line in label_file.readlines():
                    self.labels.append(line)
        except FileNotFoundError:
            logging.error('cannot find tflite label [%s]', self.label_path)
            return False
        logging.info('finished to load labels, total [%d]', len(self.labels))

        return True

    def tflite_get_label(self, index):
        """Get label string with given index.

        :param index: index for label
        :return: label string
        """
        try:
            label = self.labels[index]
        except IndexError:
            label = ''
        return label

    def update_top_label_index(self, data, data_size):
        """Update tflite label index with max score.

        :param data: array of scores
        :param data_size: data size
        :return: None
        """
        # -1 if failed to get max score index
        self.new_label_index = -1

        if data_size == len(self.labels):
            scores = [data[i] for i in range(data_size)]
            max_score = max(scores)
            if max_score > 0:
                self.new_label_index = scores.index(max_score)
                self.new_label_score = max_score
        else:
            logging.error('unexpected data size [%d]', data_size)


if __name__ == '__main__':
    od_instance = ObjectDetection(sys.argv[1:])
    od_instance.run()