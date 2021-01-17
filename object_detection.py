from argparse import ArgumentParser
import os
import gi
import cairo
import numpy as np
import time
import pandas as pd
import colorsys
import logging
import cv2

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GstVideo, GLib

from config import cfg
import gt_position_extractor
import label_mapper
import utils
import models.ssd_inception_v2.coco_labels as coco_labels


class ObjectDetector:
    """Object Detection with NNStreamer."""

    def __init__(self, argv=None):
        parser = ArgumentParser()
        parser.add_argument('--device', type=str, default=cfg.GLOBAL.DEVICE, help='device to use for inference')
        parser.add_argument('--gpu_idx', type=str, default='0', help='gpu device number to use if the gpu will be used')
        parser.add_argument('--video', type=str, default=cfg.GLOBAL.VIDEO_PATH, help='input video file path')
        parser.add_argument('--use_webcam', action='store_true', default=cfg.GLOBAL.USE_WEBCAM, help='whether use web cam or not')
        parser.add_argument('--model', type=str, choices=cfg.GLOBAL.MODELS.keys(), help='model name to use')
        parser.add_argument('--score', type=float, default=cfg.GLOBAL.SCORE_THRESHOLD, help='threshold for score')
        parser.add_argument('--train_folder', type=str, default='train', help='for train set')
        args = parser.parse_args()

        self.model_name = args.model
        self.mcfg = cfg.GLOBAL.MODELS[self.model_name]

        # set device
        self.device = args.device
        self.gpu_idx = args.gpu_idx
        if self.device == 'gpu':
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        self.framework = cfg.GLOBAL.FRAMEWORK
        self.weight_path = self.mcfg.MODEL_PATH
        self.label_path = self.mcfg.LABEL_PATH
        self.use_webcam = args.use_webcam
        self.draw_gt_box = cfg.GLOBAL.DRAW_GT
        self.file_path = args.video
        self.train_folder_path = args.train_folder

        # get video size
        vid = cv2.VideoCapture(self.file_path)
        self.video_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.model_size = self.mcfg.INPUT_SIZE

        self.box_size = 4
        self.num_labels = self.mcfg.NUM_LABELS

        # threshold values to drop detections
        self.iou_threshold = cfg.GLOBAL.IOU_THRESHOLD
        self.score_threshold = args.score

        self.loop = None
        self.pipeline = None
        self.running = False

        self.labels = {}
        self.bboxes = []
        self.times = []
        self.frames = []
        self.fps = 0
        self.avgfps = 0

        # test codes for new ssd support
        if self.model_name == 'ssd' or self.model_name == 'frcnn':
            self.labels = coco_labels.labels
        else:
            self.mapper = label_mapper.LabelMapper(self.label_path)
            self.labels = self.mapper.labels

            # load labels
            with open(self.label_path, 'r') as data:
                for ID, name in enumerate(data):
                    self.labels[ID] = name.strip('\n')

        # set colors for overlay
        hsv_tuples = [(1.0 * x / self.num_labels, 1., 1.) for x in range(self.num_labels)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        Gst.init(argv)

    # constructs model-specific gstreamer pipeline
    def construct_pipeline(self):
        self.input_dim = self.mcfg.INPUT_DIM
        self.output_dim = self.mcfg.OUTPUT_DIM
        input_type = self.mcfg.INPUT_TYPE
        output_type = self.mcfg.OUTPUT_TYPE

        video_sink = 'autovideosink' if cfg.GLOBAL.OS == 'mac' else 'ximagesink'
        enable_sync = 'true' if cfg.GLOBAL.ENABLE_SYNC else 'false'

        inputs = input_names = input_types = ''
        outputs = output_names = output_types = ''

        # input source
        if self.use_webcam:
            pipeline = 'v4l2src name=cam_src !'
        else:
            pipeline = f'filesrc location={self.file_path} ! decodebin name=decode '

        # overlay
        pipeline += f'! videoscale ! videoconvert ! video/x-raw,width={self.video_width},height={self.video_height} ! tee name=t \n' + \
                    't. ! queue ! videoconvert ! timeoverlay ! textoverlay name=text_overlay ! cairooverlay name=tensor_res ! \n' + \
                    f'     fpsdisplaysink name=fps_sink video-sink={video_sink} text-overlay=false signal-fps-measurements=true sync={enable_sync}\n'

        # tensor convert
        pipeline += f't. ! queue leaky=2 max-size-buffers={len(self.output_dim.keys())} ! videoscale add-borders=1 ! '
        if self.model_name.find('yolo') >= 0:
            pipeline += f'video/x-raw,width={self.model_size},height={self.model_size},format=RGB,pixel-aspect-ratio=1/1 ! \n' + \
            f'     tensor_converter input-dim=3:{self.model_size}:{self.model_size}:1 ! '
        elif self.model_name.find('ssd') >= 0 or self.model_name.find('rcnn') >= 0:
            pipeline += f'tensor_converter input-dim=3:{self.video_width}:{self.video_height}:1 ! '

        # tensor transform
        tensor_transform = self.mcfg.TENSOR_TRANSFORM
        if len(tensor_transform) > 0:
            pipeline += f'tensor_transform {tensor_transform} ! \n'

        # tensor filter
        # input tensors
        for input_tensor in self.input_dim:
            dim = f'{self.input_dim[input_tensor]}'
            self.input_dim[input_tensor] = dim.replace('-1:-1', f'{self.video_width}:{self.video_height}')
            inputs += f'{self.input_dim[input_tensor]},'
            input_names += f'{input_tensor},'
            input_types += f'{input_type[input_tensor]},'
        inputs = inputs[:-1]
        input_names = input_names[:-1]
        input_types = input_types[:-1]

        # output tensors
        for output_tensor in self.output_dim:
            outputs += f'{self.output_dim[output_tensor]},'
            output_names += f'{output_tensor},'
            output_types += f'{output_type[output_tensor]},'
        outputs = outputs[:-1]
        output_names = output_names[:-1]
        output_types = output_types[:-1]

        pipeline += f'\n     tensor_filter name=tensor_filter latency=1 throughput=1 framework={self.framework} model={self.weight_path} \n' + \
                    f'                   input={inputs} inputname={input_names} inputtype={input_types} \n' + \
                    f'                   output={outputs} outputname={output_names} outputtype={output_types} ! \n'

        # tensor sink
        pipeline += '     tensor_sink name=tensor_sink'

        print('', '[Pipeline]', pipeline, '', sep='\n')
        return pipeline

    def run(self):
        """Init pipeline and run example.

        :return: None
        """
        # main loop
        self.loop = GLib.MainLoop()

        pipeline = self.construct_pipeline()
        self.pipeline = Gst.parse_launch(pipeline)

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

        # frame count
        self.frame_by_bin = self.pipeline.get_by_name('decode')
        self.frame = 0

        # mesuare fps
        self.fps_sink = self.pipeline.get_by_name('fps_sink')
        self.fps_sink.connect('fps-measurements', self.on_fps_message)

        # inference performance
        self.tensor_filter = self.pipeline.get_by_name('tensor_filter')

        self.init_files()

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True
        self.start_time = time.time()

        # run main loop
        self.loop.run()

        # quit when received eos or error message
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()

        # write output
        if cfg.GLOBAL.ENABLE_PERF_LOG:
            self.csv = pd.DataFrame(self.perf_data, columns=['time', '#all_objs', '#valid_objs', 'fps', 'avgfps', 'latency', 'throughput'])
            self.csv.to_csv(self.fps_file_path, index=False, mode='w')
        
        if cfg.GLOBAL.ENABLE_DETECT_LOG:
            self.csv = pd.DataFrame(self.detected_objects_data)
            self.csv.to_csv(self.detected_objects_csv_path, index=False, header=False, mode='w')

        # calculate average
        # if len(self.frames) > 1:
        #     average_fps = (self.frames[-1] - self.frames[0]) / (self.times[-1] - self.times[0])
        #     print(f"average tensor fps: {average_fps}")

    def on_new_data(self, sink, buffer):
        """Callback for tensor sink signal.

        :param sink: tensor sink element
        :param buffer: buffer from element
        :return: None
        """

        if not self.running or buffer.n_memory() != len(self.output_dim):
            return

        success, self.frame = self.frame_by_bin.query_position(Gst.Format.DEFAULT)

        # YOLO: [1][3 * n * n][85 = 4(x_min, y_min, x_max, y_max) + 1(confidence) + 80(class scores)]
        inf_result = {}
        i = 0
        for name, dim in self.output_dim.items():
            bbox_data = utils.buffer_to_arr(self.model_name, buffer, i, utils.get_tensor_size(dim))
            inf_result[name] = bbox_data
            i += 1

        if self.model_name.find('yolo') >= 0:
            bboxes = np.reshape(np.concatenate(list(inf_result.values()), axis=0), (-1, 85))
            bboxes = utils.postprocess_boxes(bboxes, self.mcfg, self.video_width, self.video_height)
        
        elif self.model_name.find('ssd') >= 0 or self.model_name.find('rcnn') >= 0:
            bboxes = utils.decode_ssd(inf_result, self.video_width, self.video_height, self.frame)

        else:
            print(f'Error: Not supported model named {self.model_name}')
            return
        
        # bboxes, bboxes_all = self.postprocess_boxes(pred_bbox)
        # bboxes_all = self.nms(bboxes_all, method='nms')
        # for bbox in bboxes_all:
        #     self.detected_objects_data.append([str(self.frame).zfill(6), bbox[0], bbox[1], bbox[2], bbox[3], self.mapper.get_data_set_label(int(bbox[5])), bbox[4]])

        self.bboxes = utils.nms(bboxes, method='nms')

    def on_fps_message(self, fpsdisplaysink, fps, droprate, avgfps):
        self.fps = fps
        self.avgfps = avgfps
        self.droprate = droprate

        # textoverlay = self.pipeline.get_by_name('text_overlay')
        # label = 'video-fps: %.2f  overlay-fps: %.2f' % (fps, 1 / interval)
        # textoverlay.set_property('text', label)

        # perf_text = 'latency: %.3f ms  throughput: %d' % (latency, throughput)
        # textoverlay.set_property('text', perf_text)
        # print(perf_text)

    def draw_overlay_cb(self, overlay, context, timestamp, duration):
        if not self.running:
            return

        draw_cnt = 0
        context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(20)
        context.set_line_width(2.0)
        context.set_source_rgb(0.1, 1.0, 0.8)

        # bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        for detected_object in self.bboxes:
            score = detected_object[4]
            if score < 0.3:
                continue

            x = int(detected_object[0])
            y = int(detected_object[1])
            width = int(detected_object[2]) - x
            height = int(detected_object[3]) - y
            
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

            draw_cnt += 1
            # if draw_cnt >= self.max_object_detection:
            #     break

        # logging performance
        self.times.append(time.time())
        rendered_frames = self.fps_sink.get_property('frames-rendered')
        self.frames.append(rendered_frames)

        if len(self.frames) >= 2 :
            cur_time = time.time() - self.start_time
            latency = self.tensor_filter.get_property('latency') / 1000 / 10
            throughput = self.tensor_filter.get_property('throughput')
            estimated_fps = (self.frames[-1] - self.frames[-2]) / (self.times[-1] - self.times[-2])

            self.perf_data.append([cur_time, len(self.bboxes), draw_cnt, self.fps, self.avgfps, latency, throughput])

        if self.draw_gt_box:
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

    def init_files(self):
        # make out file
        folder_path = './output/'
        if cfg.GLOBAL.ENABLE_DETECT_LOG or cfg.GLOBAL.ENABLE_PERF_LOG:
            if 'train' in self.file_path:
                folder_path = folder_path + self.train_folder_path + '/' + self.file_path.split('/')[-1].split('.')[0]
            else:
                folder_path = folder_path + self.file_path.split('/')[-1].split('.')[0]

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        if self.draw_gt_box:
            self.gt_objects = {}
            
            file_path = self.file_path.split('/')[-1]
            extension_idx = file_path.rfind('.')
            file_path = file_path[:extension_idx]
            
            self.gt_objects = gt_position_extractor.GtPositionExtractor(file_path).get_gtobjects_from_csv()
            print(f'[Info] Read ground truth boxes in {len(self.gt_objects)} frames')

        if cfg.GLOBAL.ENABLE_DETECT_LOG:
            self.detected_objects_data = []

            detected_folder_path = folder_path + '/detections'
            if not os.path.exists(detected_folder_path):
                os.makedirs(detected_folder_path)

            self.detected_objects_csv_path = detected_folder_path + f'/all_detections.csv'

            i = 0
            while True:
                if not os.path.isfile(self.detected_objects_csv_path):
                    break
                else:
                    i += 1
                    if i == 1:
                        self.detected_objects_csv_path = self.detected_objects_csv_path.replace('.csv', '_1.csv')
                        if not os.path.isfile(self.detected_objects_csv_path):
                            break
                    else:
                        if f'_{i - 1}.csv' in self.detected_objects_csv_path:
                            self.detected_objects_csv_path = self.detected_objects_csv_path.replace(f'_{i - 1}.csv', f'_{i}.csv')
                        else:
                            self.detected_objects_csv_path = self.detected_objects_csv_path.replace('.csv', f'_{i}.csv')

        if cfg.GLOBAL.ENABLE_PERF_LOG:
            self.perf_data = []

            fps_folder_path = folder_path + '/fps'
            if not os.path.exists(fps_folder_path):
                os.makedirs(fps_folder_path)

            self.fps_file_path = fps_folder_path + f'/{self.model_name}_{self.score_threshold*100}.csv'

            i = 0
            while True:
                if not os.path.isfile(self.fps_file_path):
                    break
                else:
                    i += 1
                    if i == 1:
                        self.fps_file_path = self.fps_file_path.replace('.csv', '_1.csv')
                        if not os.path.isfile(self.fps_file_path):
                            break
                    else:
                        if f'_{i - 1}.csv' in self.fps_file_path:
                            self.fps_file_path = self.fps_file_path.replace(f'_{i - 1}.csv', f'_{i}.csv')
                        else:
                            self.fps_file_path = self.fps_file_path.replace('.csv', f'_{i}.csv')

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


if __name__ == '__main__':
    object_detector = ObjectDetector()
    object_detector.run()
