from easydict import EasyDict as edict

# use as from config import cfg
__C                             = edict()
cfg                             = __C

__C.GLOBAL                      = edict()
__C.SSDLITE                     = edict()
__C.YOLO_TINY                   = edict()
__C.YOLO                        = edict()

# global settings
__C.GLOBAL.MODELS               = {'ssdlite': __C.SSDLITE, 'yolo_tiny': __C.YOLO_TINY, 'yolo': __C.YOLO}
__C.GLOBAL.FRAMEWORK            = 'tensorflow'
__C.GLOBAL.DEVICE               = 'cpu'
__C.GLOBAL.VIDEO_PATH           = './video/test_video_street.mp4'
__C.GLOBAL.USE_WEBCAM           = False
__C.GLOBAL.DRAW_GT              = False
__C.GLOBAL.IOU_THRESHOLD        = 0.5
__C.GLOBAL.SCORE_THRESHOLD      = 0.3

# SSDLITE
__C.SSDLITE.MODEL_NAME         = 'ssdlite'
__C.SSDLITE.MODEL_PATH         = './models/ssdlite_v2/ssdlite_mobilenet_v2.pb'
__C.SSDLITE.LABEL_PATH         = './models/ssdlite_v2/coco_labels_list.txt'
__C.SSDLITE.INPUT_SIZE         = 300  # not used
__C.SSDLITE.NUM_LABELS         = 91   # not used
__C.SSDLITE.MAX_DETECTION      = 100
__C.SSDLITE.INPUT_DIM          = {'image_tensor': f'''3:-1:-1:1'''}
__C.SSDLITE.INPUT_TYPE         = {'image_tensor': 'uint8'}
__C.SSDLITE.OUTPUT_DIM         = {'num_detections': '1', 'detection_classes': f'''{__C.SSDLITE.MAX_DETECTION}:1''', 'detection_scores': f'''{__C.SSDLITE.MAX_DETECTION}:1''', 'detection_boxes': f'''4:{__C.SSDLITE.MAX_DETECTION}:1'''}
__C.SSDLITE.OUTPUT_TYPE        = {'num_detections': 'float32', 'detection_classes': 'float32', 'detection_scores': 'float32', 'detection_boxes': 'float32'}
__C.SSDLITE.TENSOR_TRANSFORM   = ''

# YOLO-Tiny
__C.YOLO_TINY.MODEL_NAME       = 'yolo_tiny'
__C.YOLO_TINY.MODEL_PATH       = './models/yolo_v3/yolov3_tiny.pb'
__C.YOLO_TINY.LABEL_PATH       = './models/yolo_v3/coco.names'
__C.YOLO_TINY.INPUT_SIZE       = 416
__C.YOLO_TINY.NUM_LABELS       = 80
__C.YOLO_TINY.INPUT_DIM        = {'inputs': f'''3:{__C.YOLO_TINY.INPUT_SIZE}:{__C.YOLO_TINY.INPUT_SIZE}:1'''}
__C.YOLO_TINY.INPUT_TYPE       = {'inputs': 'float32'}
__C.YOLO_TINY.OUTPUT_DIM       = {'output_boxes': '85:2535:1'}
__C.YOLO_TINY.OUTPUT_TYPE      = {'output_boxes': 'float32'}
__C.YOLO_TINY.TENSOR_TRANSFORM = 'mode=typecast option=float32'

# YOLO
__C.YOLO.MODEL_NAME             = 'yolo'
__C.YOLO.MODEL_PATH             = './models/yolo_v3/yolov3.pb'
__C.YOLO.LABEL_PATH             = './models/yolo_v3/coco.names'
__C.YOLO.INPUT_SIZE             = 416
__C.YOLO.NUM_LABELS             = 80
__C.YOLO.INPUT_DIM              = {'input/input_data': f'''3:{__C.YOLO.INPUT_SIZE}:{__C.YOLO.INPUT_SIZE}:1'''}
__C.YOLO.INPUT_TYPE             = {'input/input_data': 'float32'}
__C.YOLO.OUTPUT_DIM             = {'pred_lbbox/concat_2': '85:3:13:13:1', 'pred_mbbox/concat_2': '85:3:26:26:1', 'pred_sbbox/concat_2': '85:3:52:52:1'}
__C.YOLO.OUTPUT_TYPE            = {'pred_lbbox/concat_2': 'float32', 'pred_mbbox/concat_2': 'float32', 'pred_sbbox/concat_2': 'float32'}
__C.YOLO.TENSOR_TRANSFORM       = 'mode=arithmetic option=typecast:float32,div:255.0'
