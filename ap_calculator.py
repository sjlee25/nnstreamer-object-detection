from typing import List
import csv
from argparse import ArgumentParser

class Accuracy:
    def __init__(self, precision, recall):
        self.precision = precision
        self.recall = recall

class DetectedObject:
    def __init__(self, frame, x, y, width, height, class_id, score):
        self.frame = frame
        self.x_min = float(x)
        self.y_min = float(y)
        self.width = float(width)
        self.height = float(height)
        self.x_max = self.x_min + self.width
        self.y_max = self.y_min + self.height
        self.class_id = class_id
        self.score = score
    
    @staticmethod
    def csv_to_detected_object_list(csv_file_path):
        csv_file = open(csv_file_path, 'r', encoding='utf-8')
        # csv_file  
        ## frame, x, y, width, height, class_id, score
        csv_reader = csv.reader(csv_file)
        result = []
        for line in csv_reader :
            obj = DetectedObject(line[0],line[1],line[2],line[3],line[4],line[5],line[6])
            result.append(obj)
        return result


class ApCalculator:
    def __init__(self, accuracy_list):
        self.accuray_list = sorted(accuracy_list, key=lambda x: x.recall, reverse=True)

    def run(self):
        max_accuray_list = [0.0]*11
        for accuracy in self.accuray_list:
            for i in range(int(accuracy.recall*10)+1):
                max_accuray_list[i] = max(accuracy.precision, max_accuray_list[i])
        
        # https://seongkyun.github.io/study/2019/01/15/map/

        for i in range(int(self.accuray_list[-1].recall*10)):
            max_accuray_list[i] = 1.0
        
        print(max_accuray_list)

        return sum(max_accuray_list)/11


class AccuracyCalculator:
    def __init__(self, target_class_id_list:List[str], true_object_list:List[DetectedObject], detected_object_list:List[DetectedObject], threshold_iou = 0.5):
        self.target_class_id_list = target_class_id_list
        self.is_target_all = True if target_class_id_list == None or len(target_class_id_list) == 0 else False
        if not self.is_target_all:
            true_object_list = [obj for obj in true_object_list if obj.class_id in target_class_id_list]
            detected_object_list = [obj for obj in detected_object_list if obj.class_id in target_class_id_list]
        self.true_object_list = sorted(true_object_list, key=lambda x: x.frame)
        self.detected_object_list = sorted(detected_object_list, key=lambda x: x.frame)
        self.frame_list: List[int] = []
        for obj in true_object_list + detected_object_list:
            if obj.frame not in self.frame_list: 
                self.frame_list.append(obj.frame)
        self.threshold_iou = threshold_iou

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
        
        print(self.precision, self.recall)
        return Accuracy(self.precision, self.recall)

    def bb_intersection_over_union(self, boxA:DetectedObject, boxB:DetectedObject):
        # determine the (x, y)-coordinates of the intersection rectangle
        # box: [x_min, y_min, x_max, y_max]
        xA = max(float(boxA.x_min), float(boxB.x_min))
        yA = max(float(boxA.y_min), float(boxB.y_min))
        xB = min(float(boxA.x_max), float(boxB.x_max))
        yB = min(float(boxA.y_max), float(boxB.y_max))

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA.x_max - boxA.x_min + 1) * (boxA.y_max - boxA.y_min + 1)
        boxBArea = (boxB.x_max - boxB.x_min + 1) * (boxB.y_max - boxB.y_min + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

def calculate_accuracy(true_file_name,file_name, target_class_id_list, threshold_iou):
    true_object_file_path = true_object_file
    detected_object_file_path = file_name
    # true_object_file_path = f'./output/{true_file_name}_true_objects.csv'
    # detected_object_file_path = f'./output/{file_name}_detected_objects.csv'
    true_object_list = DetectedObject.csv_to_detected_object_list(true_object_file_path)
    detected_object_list = DetectedObject.csv_to_detected_object_list(detected_object_file_path)
    cal = AccuracyCalculator(target_class_id_list,true_object_list,detected_object_list,threshold_iou=threshold_iou)
    accuracy = cal.run()
    return accuracy

    
if __name__ == "__main__":
    
    #    USAGE
    # accuracy_list = [Accuracy(0.5,0.4), Accuracy(0.5,0.5), Accuracy(0.4,0.9)]
    # calculator = ApCalculator(accuracy_list)
    # result = calculator.run()
    
    parser = ArgumentParser()
    parser.add_argument('--true_object_file', type=str, help="enter a csv file name contains true objects")
    parser.add_argument('--folder_path', type=str, help="enter a folder path contains csv files")
    parser.add_argument('--target_class_id_list', type=str, help="enter a list of target class ids seperated by ','")
    parser.add_argument('--threshold_iou', type=float,)

    args = parser.parse_args()

    true_object_file = args.true_object_file
    folder_path = args.folder_path
    target_class_id_list = [] 
    if args.target_class_id_list:
        target_class_id_list = args.target_class_id_list.split(',')
    threshold_iou = args.threshold_iou


    accuracy_list:List[Accuracy] = []
    file_number = 0
    try :
        while True:
            file_name = folder_path + '/' + str(file_number) + '.csv'
            file_number += 1
            accuracy = calculate_accuracy(true_object_file, file_name, target_class_id_list, threshold_iou=threshold_iou)
            accuracy_list.append(accuracy)
    except Exception as ex:
        print(ex)
        calculator = ApCalculator(accuracy_list)
        result = calculator.run()

        print(result)
