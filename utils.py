from gi.repository import Gst
import numpy as np
import cairo

from config import cfg


def get_tensor_size(dim):
    dims = list(map(int, dim.split(':')))
    return np.prod(dims)


def buffer_to_arr(model_name, buffer, idx, expected_size, data_type=np.float32):
    buffer_content = buffer.get_memory(idx)
    result, mapinfo_content = buffer_content.map(Gst.MapFlags.READ)

    if result:
        if mapinfo_content.size == expected_size * 4:
            content_arr = np.frombuffer(mapinfo_content.data, dtype=data_type)
            buffer_content.unmap(mapinfo_content)

            if model_name.find('yolo') > 0:
                return np.reshape(content_arr, (-1, 85))
            else:
                return content_arr

    else:
        buffer_content.unmap(mapinfo_content)
        print(f'''Error: getting memory from buffer with index {idx} failed''')
        exit(1)


def decode_ssd(inf_result, video_width, video_height, frame):
    scores = inf_result['detection_scores']
    classes = inf_result['detection_classes']
    boxes = np.reshape(inf_result['detection_boxes'], (-1, 4))
    # num_detections = int(inf_result['num_detections'])

    coors = boxes.copy()
    coors[:, [0, 1]] = coors[:, [1, 0]]
    coors[:, [2, 3]] = coors[:, [3, 2]]
    coors[:, 0::2] *= video_width
    coors[:, 1::2] *= video_height

    mask = scores >= cfg.GLOBAL.SCORE_THRESHOLD
    coors, scores, classes = coors[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    # while added_objects < num_detections:
    #     added_objects += 1
    #     idx += 1
    #     score = scores[idx]
    #     label_idx = int(classes[idx])

    #     # [y_min, x_min, y_max, x_max]
    #     box_idx = 4 * idx
    #     x_min = boxes[box_idx + 1]
    #     x_max = boxes[box_idx + 3]
    #     y_min = boxes[box_idx + 0]
    #     y_max = boxes[box_idx + 2]

    #     detected_objects.append([x_min, y_min, x_max, y_max, score, label_idx])

    # return np.array(detected_objects)


def iou(boxes1, boxes2):
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

def nms(bboxes, sigma=0.3, method='nms'):
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
            iou_val = iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou_val),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou_val > cfg.GLOBAL.IOU_THRESHOLD
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou_val ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox, model_cfg, video_width, video_height):
    # pred_bbox = np.array(pred_bbox)
    valid_scale = [0, np.inf]
    model_size = model_cfg.INPUT_SIZE
    model_name = model_cfg.MODEL_NAME

    resize_ratio = min(model_size / video_width, model_size / video_height)
    dw = (model_size - resize_ratio * video_width) / 2.
    dh = (model_size - resize_ratio * video_height) / 2.

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    if model_name == 'yolo_tiny':
        pred_coor = np.array(pred_xywh)
    else: pred_coor = np.concatenate(
        [pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5, pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [video_width - 1, video_height - 1])],
                               axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    # pred_coor = np.array(pred_coor, dtype=np.uint16)

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores >= cfg.GLOBAL.SCORE_THRESHOLD
    # score_mask_all = scores > 0.

    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
    decoded_bbox = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    # mask_all = np.logical_and(scale_mask, score_mask_all)
    # coors_all, scores_all, classes_all = pred_coor[mask_all], scores[mask_all], classes[mask_all]
    # decoded_bbox_all = np.concatenate([coors_all, scores_all[:, np.newaxis], classes_all[:, np.newaxis]], axis=-1)

    # return decoded_bbox, decoded_bbox_all
    return decoded_bbox
