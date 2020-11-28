from argparse import ArgumentParser
import os
import gi
import cairo
import numpy as np
import time
import pandas as pd
import colorsys
import cv2  # for temporarily get video size

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GstVideo, GLib

from config import cfg
import gt_position_extractor
import label_mapper
import utils


class ObjectDetector:
    """Object Detection with NNStreamer."""

    def __init__(self, argv=None):
        parser = ArgumentParser()
        parser.add_argument('--device', type=str, default=cfg.GLOBAL.DEVICE, help='device to use for inference')
        parser.add_argument('--gpu_idx', type=str, default='0', help='gpu device number to use if the gpu will be used')
        parser.add_argument('--video', type=str, default=cfg.GLOBAL.VIDEO_PATH, help='input video file path')
        parser.add_argument('--use_webcam', action='store_true', default=cfg.GLOBAL.USE_WEBCAM,
                            help='whether use web cam or not')
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

        # cairo overlay state
        # self.cairo_valid = True
        self.valid_scale = [0, np.inf]

        # load labels
        with open(self.label_path, 'r') as data:
            for ID, name in enumerate(data):
                self.labels[ID] = name.strip('\n')

        self.mapper = label_mapper.LabelMapper(self.od_label)
        self.labels = self.mapper.labels

        # set colors for overlay
        hsv_tuples = [(1.0 * x / self.num_labels, 1., 1.) for x in range(self.num_labels)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        Gst.init(argv)

    # constructs model-specific gstreamer pipeline
    def construct_pipeline(self):
        self.input_dim = self.mcfg.INPUT_DIM
        input_type = self.mcfg.INPUT_TYPE
        self.output_dim = self.mcfg.OUTPUT_DIM
        output_type = self.mcfg.OUTPUT_TYPE

        inputs = input_names = input_types = ''
        outputs = output_names = output_types = ''

        # input source
        if self.use_webcam:
            pipeline = 'v4l2src name=cam_src ! videoscale '
        else:
            pipeline = f'''filesrc location={self.file_path} ! decodebin name=decode decode. ! videoscale ! videorate '''

        # overlay
        pipeline += '! videoconvert ! timeoverlay ! textoverlay name=text_overlay ! \n' + \
                    '''     video/x-raw,width={self.video_width},height={self.video_height},format=RGB ! tee name=t \n''' + \
                    't. ! queue ! videoconvert ! cairooverlay name=tensor_res tensor_res. ! ' + \
                    'fpsdisplaysink name=fps_sink video-sink=ximagesink text-overlay=false signal-fps-measurements=true \n'

        # tensor convert
        pipeline += f'''t. ! queue ! leaky=2 max-size-buffers={len(output_dim.keys())} ! videoscale add-borders=1 ! ''' + \
                    f'''video/x-raw,width={self.model_size},height={self.model_size},format=RGB,pixel-aspect-ratio=1/1 ! \n''' + \
                    f'''     tensor_converter input-dim=3:{self.model_size}:{self.model_size}:1 ! '''

        # tensor transform
        tensor_transform = self.mcfg.TENSOR_TRANSFORM
        if len(tensor_transform) > 0:
            pipeline += f'''tensor_transform {tensor_transform} ! \n'''

        # tensor filter
        # input tensors
        for input_tensor in input_dim:
            inputs += f'''{input_dim[input_tensor]},'''
            input_names += f'''{input_tensor},'''
            input_types += f'''{input_type[input_tensor]},'''
        inputs = inputs[:-1]
        input_names = input_names[:-1]
        input_types = input_types[:-1]

        # output tensors
        for output_tensor in output_dim:
            outputs += f'''{output_dim[output_tensor]},'''
            output_names += f'''{output_tensor},'''
            output_types += f'''{output_type[output_tensor]},'''
        outputs = outputs[:-1]
        output_names = output_names[:-1]
        output_types = output_types[:-1]

        pipeline += f'''     tensor_filter framework={self.framework} model={self.weight_path} \n''' + \
                    f'''                   input={inputs} inputname={input_names} inputtype={input_types} \n''' + \
                    f'''                   output={outputs} outputname={output_names} outputtype={output_types} \n'''

        # tensor sink
        pipeline += '     tensor_sink name=tensor_sink'

        print('[Pipeline]', pipeline, '', sep='\n')
        return pipeline

    def run(self):
        """Init pipeline and run example.

        :return: None
        """
        # main loop
        self.loop = GLib.MainLoop()

        pipeline = self.construct_pipeline()
        self.pipeline = Gst.parse_launch(pipeline)

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
        fps_sink = self.pipeline.get_by_name('fps_sink')
        fps_sink.connect('fps-measurements', self.on_fps_message)

        self.init_files()

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        # run main loop
        self.loop.run()

        # quit when received eos or error message
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)

        # write output
        self.csv = pd.DataFrame(self.fps_data, columns=['fps', 'tensor_fps'])
        self.csv.to_csv(fps_file_path, index=False, mode='w')
        self.csv = pd.DataFrame(self.detected_objects_data)
        self.csv.to_csv(detected_objects_csv_path, index=False, header=False, mode='w')

        # calculate average
        interval = (self.times[-1] - self.times[0])
        print(f"average tensor fps: {(len(self.times) - 1) / interval}")

    def on_new_data(self, sink, buffer):
        """Callback for tensor sink signal.

        :param sink: tensor sink element
        :param buffer: buffer from element
        :return: None
        """

        if not self.running or buffer.n_memory() != 3:
            return

        # YOLO: [1][3 * n * n][85 = 4(x_min, y_min, x_max, y_max) + 1(confidence) + 80(class scores)]
        pred_bboxes = {}
        i = 0
        for name, dim in self.output_dim.items():
            pred_bboxes[name] = self.get_arr_from_buffer(self.model_name, buffer, i, utils.get_tensor_size(dim))
            i += 1
        # pred_lbbox = self.get_arr_from_buffer(self.model_name, buffer, 0, 13 * 13 * 3 * 85)
        # pred_mbbox = self.get_arr_from_buffer(self.model_name, buffer, 1, 26 * 26 * 3 * 85)
        # pred_sbbox = self.get_arr_from_buffer(self.model_name, buffer, 2, 52 * 52 * 3 * 85)

        pred_bbox = np.concatenate([bbox for bbox in pred_bboxes.values()], axis=0)
        success, self.frame = self.frame_by_bin.query_position(Gst.Format.DEFAULT)

        bboxes = self.postprocess_boxes(pred_bbox)
        # bboxes, bboxes_all = self.postprocess_boxes(pred_bbox)
        # bboxes_all = self.nms(bboxes_all, method='nms')
        # for bbox in bboxes_all:
        #     self.detected_objects_data.append([str(self.frame).zfill(6), bbox[0], bbox[1], bbox[2], bbox[3], self.mapper.get_data_set_label(int(bbox[5])), bbox[4]])

        self.bboxes = self.nms(bboxes, method='nms')
        self.times.append(time.time())

    def draw_overlay_cb(self, overlay, context, timestamp, duration):
        if not self.running:
            return

        # draw_cnt = 0
        context.select_font_face('Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_font_size(20)
        context.set_line_width(2.0)
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

    def on_fps_message(self, fpsdisplaysink, fps, droprate, avgfps):
        if len(self.times) >= 2:
            interval = self.times[-1] - self.times[-2]
            label = 'video-fps: %.2f  overlay-fps: %.2f' % (fps, 1 / interval)
            textoverlay = self.pipeline.get_by_name('text_overlay')
            textoverlay.set_property('text', label)
            self.fps_data.append([fps, 1 / interval])

    def init_files(self):
        # make out file
        folder_path = './output/'
        if 'train' in self.file_path:
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
                    if f'_{i - 1}.csv' in detected_objects_csv_path:
                        detected_objects_csv_path = detected_objects_csv_path.replace(f'_{i - 1}.csv', f'_{i}.csv')
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
                    if f'_{i - 1}.csv' in fps_file_path:
                        fps_file_path = fps_file_path.replace(f'_{i - 1}.csv', f'_{i}.csv')
                    else:
                        fps_file_path = fps_file_path.replace('.csv', f'_{i}.csv')

        self.detected_objects_data = []
        self.fps_data = []

        self.gt_objects = {}
        # self.draw_gt_box = True
        self.draw_gt_box = cfg.GLOBAL.DRAW_GT
        if self.draw_gt_box:
            file_path = self.file_path.split('/')[-1]
            extension_idx = file_path.rfind('.')
            file_path = file_path[:extension_idx]
            self.gt_objects = gt_position_extractor.GtPositionExtractor(file_path).get_gtobjects_from_csv()
            print(f'''[Info] Read ground truth boxes in {len(self.gt_objects)} frames''')


if __name__ == '__main__':
    object_detector = ObjectDetector()
    object_detector.run()
