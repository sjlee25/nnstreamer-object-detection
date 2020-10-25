from typing import List

class Accuracy:
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall

class DetectedObject:
    def __init__(self, frame:int, x, y, width, height, class_id, score):
        self.x_min = x
        self.y_min = y
        self.width = width
        self.height = height
        self.x_max = x + width
        self.y_max = y + height
        self.class_id = class_id
        self.score = score


class ApCalculator:
    def __init__(self, accuracy_list:List[Accuracy]):
        self.accuray_list = sorted(accuracy_list, key=lambda x: x.recall, reverse=True)

    def run(self):
        max_accuray_list = [0.0]*11
        for accuracy in self.accuray_list:
            for i in range(int(accuracy.recall*10)+1):
                max_accuray_list[i] = max(accuracy.precision, max_accuray_list[i])
        
        # 0.0부터 가장 작은 recall값의 accuray가 주어질 때까지는 1.0으로 변경
        # https://seongkyun.github.io/study/2019/01/15/map/

        for i in range(int(self.accuray_list[-1].recall*10)):
            max_accuray_list[i] = 1.0
        
        print(max_accuray_list)

        return sum(max_accuray_list)/11


class AccuracyCalculator:
    def __init__(self, target_class_id_list:List[str], true_object_list:List[DetectedObject], detected_object_list:List[DetectedObject]):
        self.target_class_id_list = target_class_id_list
        self.is_target_all = True if target_class_id_list == None or len(target_class_id_list) == 0 else False
        if not self.is_target_all:
            true_object_list = [obj for obj in true_object_list if obj.class_id in target_class_id_list]
            detected_object_list = [obj for obj in detected_object_list if obj.class_id in target_class_id_list]
        self.true_object_list = sorted(true_object_list, key=lambda x: x.frame)
        self.detected_object_list = sorted(detected_object_list, key=lambda x: x.frame)
        self.frame_list: List[int] = []
        for obj in true_object_list + detected_object_list:
            if not(obj.frame in self.frame_list): 
                self.frame_list.append(obj.frame)
        self.threshold_iou = 0.5

    def run(self):
        self.true_detected_count = 0

        for frame in self.frame_list:
            true_object_list = [obj for obj in self.true_object_list if obj.frame == frame]
            detected_object_list = [obj for obj in self.detected_object_list if obj.frame == frame]
            for true_object in true_object_list:
                for detected_object in detected_object_list:
                    iou = self.bb_intersection_over_union(true_object, detected_object)
                    if iou >= self.threshold_iou:
                        self.true_detected_count = self.true_detected_count + 1

        
        self.precision = self.true_detected_count/len(self.detected_object_list) if len(self.detected_object_list) != 0 else 1.0
        self.recall = self.true_detected_count/len(self.true_object_list) if len(self.true_object_list) != 0 else 1.0
        
        return self.precision, self.recall

    def bb_intersection_over_union(self, boxA:DetectedObject, boxB:DetectedObject):
        # determine the (x, y)-coordinates of the intersection rectangle
        # box: [x_min, y_min, x_max, y_max]
        xA = max(boxA.x_min, boxB.x_min)
        yA = max(boxA.y_min, boxB.y_min)
        xB = min(boxA.x_max, boxB.x_max)
        yB = min(boxA.y_max, boxB.y_max)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
    

    
if __name__ == "__main__":
    accuracy_list = [Accuracy(0.5,0.4), Accuracy(0.5,0.5), Accuracy(0.4,0.9)]
    calculator = ApCalculator(accuracy_list)
    print(calculator.run())