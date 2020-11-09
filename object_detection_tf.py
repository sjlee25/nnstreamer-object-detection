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

from argparse import ArgumentParser
import os
import sys
import logging
import gi
from math import exp
import cairo
import struct
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
        self.od_model = args.model if args.model else './models/ssdlite_v2/ssdlite_mobilenet_v2.pb'
        self.od_label = args.label if args.label else './models/ssdlite_v2/coco_labels_list.txt'
        self.use_web_cam = args.use_web_cam if args.use_web_cam else False
        self.file_path = args.file if args.file else './video/test_video_street.mp4'
        self.train_folder_path = args.train_folder if args.train_folder else 'train'
        
        self.loop = None
        self.pipeline = None
        self.running = False
        
        self.labels = []
        self.detected_objects = []
        self.times = []

        vid = cv2.VideoCapture(self.file_path)
        self.video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.box_size = 4
        # self.label_size = 91
        self.detection_max = 100
        
        # max objects in display
        # self.max_object_detection = 20

        # threshold values to drop detections
        self.threshold_iou = 0.5
        self.threshold_score = 0.3
        # self.threshold_score = args.threshold_score if args.threshold_score else self.threshold_score

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

        # gstreamer pipeline
        if self.use_web_cam :
            pipeline = 'v4l2src name=cam_src ! videoscale '
        else :
            pipeline = f'filesrc location={self.file_path} ! decodebin name=decode decode. ! videoscale ! videorate '

        pipeline += f'''
            ! videoconvert ! timeoverlay ! textoverlay name=text_overlay ! video/x-raw,width={self.video_width},height={self.video_height},format=RGB ! tee name=t  
            t. ! queue ! videoconvert ! cairooverlay name=tensor_res tensor_res. ! fpsdisplaysink name=fps_sink video-sink=ximagesink text-overlay=false signal-fps-measurements=true 
            t. ! queue leaky=2 max-size-buffers=4 ! videoscale ! tensor_converter ! 
                tensor_filter framework={self.od_framework} model={self.od_model} 
                    input=3:{self.video_width}:{self.video_height}:1 inputname=image_tensor inputtype=uint8 
                    output=1,{self.detection_max}:1,{self.detection_max}:1,{self.box_size}:{self.detection_max}:1  
                    outputname=num_detections,detection_classes,detection_scores,detection_boxes 
                    outputtype=float32,float32,float32,float32 ! 
                tensor_sink name=tensor_sink
        '''
        self.pipeline = Gst.parse_launch(pipeline)
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
        self.set_window_title('img_tensor', 'Object Detection with SSDLite')

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

        fps_file_path = fps_folder_path + f'/{self.threshold_score}.csv'

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
        print(f"Average overlay-fps: {(len(self.times)-1)/interval}")

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
    def nms(self, detected_objs, frame):
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
                obj_score = detected_objs[i].score
                if obj_score > 0.:
                    self.detected_objects_data.append([frame, detected_objs[i].x, detected_objs[i].y, detected_objs[i].width, detected_objs[i].height, self.mapper.get_data_set_label(detected_objs[i].class_id), detected_objs[i].score])
                if obj_score >= self.threshold_score:
                    self.detected_objects.append(detected_objs[i])
        
    def get_detected_objects(self, num_detections, classes, scores, boxes, frame):
        detected = []
        added_objects = 0
        idx = -1

        while added_objects < num_detections:
            added_objects += 1
            idx += 1
            obj_score = scores[idx]
            obj_class = int(classes[idx])

            # [y_min, x_min, y_max, x_max]
            box_idx = self.box_size * idx
            x_min = boxes[box_idx +  1]
            x_max = boxes[box_idx + 3]
            y_min = boxes[box_idx + 0]
            y_max = boxes[box_idx + 2]

            x = x_min * self.video_width
            y = y_min * self.video_height
            width = (x_max - x_min) * self.video_width
            height = (y_max - y_min) * self.video_height

            detected.append(DetectedObject(x, y, width, height, obj_class, obj_score))

        self.nms(detected,frame)

    def get_arr_from_buffer(self, buffer, idx, expected_size, get_type):
        buffer_content = buffer.get_memory(idx)
        result, mapinfo_content = buffer_content.map(Gst.MapFlags.READ)
        
        if result:
            if mapinfo_content.size == expected_size:
                content_arr = struct.unpack(str(expected_size//4)+get_type, mapinfo_content.data)
                buffer_content.unmap(mapinfo_content)
                return content_arr
        else:
            print('Error: getting memory from buffer with index %d failed' % (idx))
            buffer_content.unmap(mapinfo_content)
            exit(1)

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
            
            success, self.frame = self.frame_by_bin.query_position(Gst.Format.DEFAULT)

            self.get_detected_objects(num_detections, detection_classes, detection_scores, detection_boxes, str(self.frame).zfill(6))
            self.times.append(time.time())

    # def prepare_overlay_cb(self, overlay, caps):
    #     # print(self, overlay, caps)
    #     # self.cairo_valid = GstVideo.VideoInfo.from_caps(caps)
    #     self.cairo_valid = True

    def draw_overlay_cb(self, overlay, context, timestamp, duration):      
        if not self.cairo_valid or not self.running:
            return
        
        # draw_cnt = 0
        context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(20)
        context.set_line_width(2.0)
        context.set_source_rgb(0.1, 1.0, 0.8)

        for detected_object in self.detected_objects:
            x = detected_object.x
            y = detected_object.y
            width = detected_object.width
            height = detected_object.height

            label = self.get_label(detected_object.class_id)
            score = detected_object.score

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

        self.mapper = label_mapper.LabelMapper(self.od_label)
        self.labels = self.mapper.labels
        # try:
        #     with open(self.od_label, 'r') as label_file:
        #         for line in label_file.readlines():
        #             if line[-1] == '\n':
        #                 line = line[:-1]
        #             self.labels.append(line)
        # except FileNotFoundError:
        #     logging.error('cannot find tflite label [%s]', self.od_label)
        #     return False
        logging.info('finished to load labels, total [%d]', len(self.labels))

        return True

    def get_label(self, index):
        """Get label string with given index.

        :param index: index for label
        :return: label string
        """
        try:
            label = self.labels[index]
        except IndexError:
            label = ''
        return label

    def on_fps_message(self, fpsdisplaysink, fps, droprate, avgfps):
        if len(self.times) >= 2:
            interval = self.times[-1] - self.times[-2]
            label = 'video-fps: %.2f  overlay-fps: %.2f' % (fps, 1/interval)
            textoverlay = self.pipeline.get_by_name('text_overlay')
            textoverlay.set_property('text', label)
            self.fps_data.append([fps, 1/interval])
            # print(f'{fps} {1/interval}')

if __name__ == '__main__':
    od_instance = ObjectDetection()
    od_instance.run()