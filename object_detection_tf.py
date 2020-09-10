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
$ python3 object_detection_tf.py 

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
        if len(sys.argv) != 4:
            print('usage: python3 filename.py [folder path] [model file] [label file]')
            exit(1)

        self.od_framework= 'tensorflow'
        self.file_path = './' + sys.argv[1] + '/'
        self.od_model = self.file_path + sys.argv[2]
        self.od_label = self.file_path + sys.argv[3]

        self.loop = None
        self.pipeline = None
        self.running = False
        
        self.labels = []
        self.detected_objects = []

        self.video_width = 640
        self.video_height = 480
        self.model_width = 640
        self.model_height = 480

        self.box_size = 4
        # self.label_size = 91
        self.detection_max = 100
        
        # max objects in display
        self.max_object_detection = 20

        # threshold values to drop detections
        self.threshold_iou = 0.5
        self.threshold_score = 0.3

        # cairo overlay state
        self.cairo_valid = True

        if not self.model_init():
            raise Exception

        GObject.threads_init()
        Gst.init(argv)

    def run(self):
        """Init pipeline and run example.

        :return: None
        """
        # main loop
        self.loop = GObject.MainLoop()

        # new tf pipeline
        self.pipeline = Gst.parse_launch(
            'v4l2src name=cam_src ! videoscale ! videoconvert ! video/x-raw,width=%d,height=%d,format=RGB,framerate=30/1 ! tee name=t ' % (self.video_width, self.video_height) +
            't. ! queue ! videoconvert ! cairooverlay name=tensor_res ! ximagesink name=img_tensor '
            't. ! queue leaky=2 max-size-buffers=4 ! videoscale ! tensor_converter ! '
                'tensor_filter framework=%s model=%s ' % (self.od_framework, self.od_model) +
                    'input=3:%d:%d:1 inputname=image_tensor inputtype=uint8 ' % (self.video_width, self.video_height) +
                    'output=1,%d:1,%d:1,%d:%d:1 ' % (self.detection_max, self.detection_max, self.box_size, self.detection_max) +
                    'outputname=num_detections,detection_classes,detection_scores,detection_boxes '
                    'outputtype=float32,float32,float32,float32 ! '
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
        # overlay.connect('caps-changed', self.prepare_overlay_cb)

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

        return max(ratio, 0)

    # pass only one box which has higher probability out of two
    # if their iou value is higher than the threshold
    def nms(self, detected_objs):
        num_boxes = len(detected_objs)
        detected_objs = sorted(detected_objs, key=lambda obj: obj.score, reverse=True)
        to_delete = [False] * num_boxes

        for i in range(num_boxes):
            if to_delete[i]: continue
            for j in range(i + 1, num_boxes):
                if self.iou(detected_objs[i], detected_objs[j]) > self.threshold_iou:
                    to_delete[j] = True

        self.detected_objects.clear()
        for i in range(num_boxes):
            if not to_delete[i]:
                self.detected_objects.append(detected_objs[i])

    def get_detected_objects(self, num_detections, classes, scores, boxes):
        detected = []
        added_objects = 0
        idx = -1

        while added_objects < num_detections:
            idx += 1
            obj_score = scores[idx]
            if obj_score < self.threshold_score: continue
            obj_class = int(classes[idx])

            box_idx = self.box_size * idx
            x_min = boxes[box_idx + 0]
            x_max = boxes[box_idx + 1]
            y_min = boxes[box_idx + 2]
            y_max = boxes[box_idx + 3]

            x = x_min * self.model_width
            y = y_min * self.model_height
            width = (x_max - x_min) * self.model_width
            height = (y_max - y_min) * self.model_height

            detected.append(DetectedObject(x, y, width, height, obj_class, obj_score))
            added_objects += 1

        self.nms(detected)

    def get_arr_from_buffer(self, buffer, idx, expected_size, get_type):
        buffer_content = buffer.get_memory(idx)
        result, mapinfo_content = buffer_content.map(Gst.MapFlags.READ)
        
        if result:
            if mapinfo_content.size == expected_size:
                content_arr = struct.unpack(str(expected_size//4)+get_type, mapinfo_content.data)
        else:
            print('Error: getting memory from buffer with index %d failed' % (idx))
            exit(1)
        
        return content_arr

    def on_new_data(self, sink, buffer):
        """Callback for tensor sink signal.

        :param sink: tensor sink element
        :param buffer: buffer from element
        :return: None
        """

        if self.running and buffer.n_memory() == 4:
            num_detections = int(self.get_arr_from_buffer(buffer, 0, 4, 'f')[0])                                                            # 1
            detection_classes = self.get_arr_from_buffer(buffer, 1, self.detection_max * 4, 'f')                             # 100
            detection_scores = self.get_arr_from_buffer(buffer, 2, self.detection_max * 4, 'f')                              # 100
            detection_boxes = self.get_arr_from_buffer(buffer, 3, self.box_size * self.detection_max * 4, 'f')  # 400
        
            self.get_detected_objects(num_detections, detection_classes, detection_scores, detection_boxes)

    # def prepare_overlay_cb(self, overlay, caps):
    #     # print(self, overlay, caps)
    #     # self.cairo_valid = GstVideo.VideoInfo.from_caps(caps)
    #     self.cairo_valid = True

    def draw_overlay_cb(self, overlay, context, timestamp, duration):      
        if not self.cairo_valid or not self.running:
            return
        
        draw_cnt = 0
        # context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(20)

        for detected_object in self.detected_objects:
            label = self.labels[detected_object.class_id]
            
            x = detected_object.x * self.video_width / self.model_width
            y = detected_object.y * self.video_height / self.model_height
            width = detected_object.width * self.video_width / self.model_width
            height = detected_object.height * self.video_height / self.model_height

            # draw rectangle
            context.rectangle(x, y, width, height)
            context.set_source_rgb(0, 1, 1)
            context.set_line_width(2.0)
            context.stroke()
            context.fill_preserve()

            # draw title
            context.move_to(x, y - 10)
            context.text_path('%s  %.2f' % (label, detected_object.score))
            context.set_source_rgb(0, 1, 1)
            context.fill_preserve()
            context.set_source_rgb(1, 1, 1)
            context.set_line_width(0.3)
            context.stroke()
            context.fill_preserve()

            draw_cnt += 1
            if draw_cnt >= self.max_object_detection:
                break

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

    def model_init(self):
        """Check tflite model and load labels.

        :return: True if successfully initialized
        """
        # od_model = 'ssd_mobilenet_v2_coco.tflite'
        # od_label = 'coco_labels_list.txt'
        # od_box = 'box_priors.txt'
        # current_folder = os.path.dirname(os.path.abspath(__file__))
        # model_folder = os.path.join(current_folder, 'od_model')

        # check model file exists
        # self.od_model = os.path.join(model_folder, od_model)
        # if not os.path.exists(self.od_model):
        #     logging.error('cannot find tflite model [%s]', self.od_model)
        #     return False

        # load labels
        # self.od_label = os.path.join(model_folder, od_label)
        try:
            with open(self.od_label, 'r') as label_file:
                for line in label_file.readlines():
                    if line[-1] == '\n':
                        line = line[:-1]
                    self.labels.append(line)
        except FileNotFoundError:
            logging.error('cannot find tflite label [%s]', self.od_label)
            return False
        logging.info('finished to load labels, total [%d]', len(self.labels))

        return True

    def get_od_label(self, index):
        """Get label string with given index.

        :param index: index for label
        :return: label string
        """
        try:
            label = self.labels[index]
        except IndexError:
            label = ''
        return label

if __name__ == '__main__':
    od_instance = ObjectDetection()
    od_instance.run()