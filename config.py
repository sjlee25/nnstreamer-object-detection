from easydict import EasyDict as edict

# use as from config import cfg
__C                         = edict()
cfg                         = __C

__C.GLOBAL                  = edict()
__C.SSD_LITE                = edict()
__C.YOLO_TINY               = edict()
__C.YOLO                    = edict()

# global settings
__C.GLOBAL.MODELS           = {'ssd_lite': __C.SSD_LITE, 'yolo_tiny': __C.YOLO_TINY, 'yolo': __C.YOLO}
__C.GLOBAL.FRAMEWORK        = 'tensorflow'
__C.GLOBAL.DEVICE           = 'cpu'
__C.GLOBAL.VIDEO_PATH       = './video/test_video_street.mp4'
__C.GLOBAL.USE_WEBCAM       = False
__C.GLOBAL.IOU_THRESHOLD    = 0.5
__C.GLOBAL.SCORE_THRESHOLD  = 0.3
__C.GLOBAL.DRAW_GT          = False

# YOLO
__C.YOLO.MODEL_PATH         = './models/yolo_v3/yolov3.pb'
__C.YOLO.LABEL_PATH         = './models/yolo_v3/coco.names'
__C.YOLO.INPUT_SIZE         = 416
__C.YOLO.NUM_LABELS         = 80
__C.YOLO.INPUT_DIM          = {'input/input_data': f'''3:{__C.YOLO.INPUT_SIZE}:{__C.YOLO.INPUT_SIZE}:1'''}
__C.YOLO.INPUT_TYPE         = {'input/input_data': 'float32'}
__C.YOLO.OUTPUT_DIM         = {'pred_lbbox/concat_2': '85:3:13:13:1', 'pred_mbbox/concat_2': '85:3:26:26:1', 'pred_sbbox/concat_2': '85:3:52:52:1'}
__C.YOLO.OUTPUT_TYPE        = {'pred_lbbox/concat_2': 'float32', 'pred_mbbox/concat_2': 'float32', 'pred_sbbox/concat_2': 'float32'}
__C.YOLO.TENSOR_TRANSFORM   = 'mode=arithmetic option=typecast:float32,div:255.0'
