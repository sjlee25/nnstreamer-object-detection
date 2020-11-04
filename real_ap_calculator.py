from typing import List
import csv
from argparse import ArgumentParser
import os
import pandas as pd

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
        self.accuray_list = accuracy_list

    def run(self):
        class_id_list:List[str] = []
        for accuracy in self.accuray_list:
            for key in accuracy:
                if key not in class_id_list:
                    class_id_list.append(key)
        
        result_data = []
        for class_id in class_id_list:
            print(f'calculating {class_id}...')
            ap = self.calculate(class_id)
            result_data.append([class_id, ap])
        
        map_sum = 0
        for result in result_data:
            map_sum += result[1]
        result_data.insert(0,['map', map_sum/len(result_data)])

        print('==========================================')
        print('# result')
        for result in result_data:
            print(f'{result[0]}: {result[1]}')

        result_folder_path = './result'
        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)
        
        result_file_path = result_folder_path + '/map.csv'
        i = 0
        while True:
            if not os.path.isfile(result_file_path):
                break
            else:
                i += 1
                if i == 1:
                    result_file_path = result_file_path.replace('.csv', '_1.csv')
                    if not os.path.isfile(result_file_path):
                        break
                else:
                    if f'_{i-1}.csv' in result_file_path:
                        result_file_path = result_file_path.replace(f'_{i-1}.csv', f'_{i}.csv')
                    else:
                        result_file_path = result_file_path.replace('.csv', f'_{i}.csv')
        
        csv = pd.DataFrame(result_data)
        csv.to_csv(result_file_path, index=False, header=False, mode='w')
                   
    def calculate(self, class_id):
        accuracy_list = [obj[class_id] for obj in self.accuray_list]
        accuracy_list = sorted(accuracy_list, key=lambda x: x.recall, reverse=True)
        max_accuray_list = [0.0]*11
        for accuracy in accuracy_list:
            for i in range(int(accuracy.recall*10)+1):
                max_accuray_list[i] = max(accuracy.precision, max_accuray_list[i])
        
        # https://seongkyun.github.io/study/2019/01/15/map/

        for i in range(int(accuracy_list[-1].recall*10)):
            max_accuray_list[i] = 1.0

        return sum(max_accuray_list)/11



class AccuracyCalculator:
    def __init__(self, true_object_list:List[DetectedObject], detected_object_list:List[DetectedObject], threshold_iou = 0.5):
        self.true_object_list = sorted(true_object_list, key=lambda x: x.frame)
        self.detected_object_list = sorted(detected_object_list, key=lambda x: x.frame)
        self.threshold_iou = threshold_iou

    def run(self):
        class_id_list: List[str] = []

        for obj in self.true_object_list:
            if obj.class_id not in class_id_list:
                class_id_list.append(obj.class_id)

        result = {}
        for class_id in class_id_list:
            accuracy = self.calculate(class_id)
            result[class_id] = accuracy

        return result
    
    def calculate(self, class_id):
        true_object_list = [obj for obj in self.true_object_list if obj.class_id == class_id]
        detected_object_list = [obj for obj in self.detected_object_list if obj.class_id == class_id]

        frame_list: List[int] = []
        for obj in true_object_list:
            if obj.frame not in frame_list:
                frame_list.append(obj.frame)

        true_detected_count = 0

        for frame in frame_list:
            frame_true_object_list = [obj for obj in true_object_list if obj.frame == frame]
            frame_detected_object_list = [obj for obj in detected_object_list if obj.frame == frame]
            for true_object in frame_true_object_list:
                for detected_object in frame_detected_object_list:
                    iou = self.bb_intersection_over_union(true_object, detected_object)
                    if iou >= self.threshold_iou:
                        true_detected_count = true_detected_count + 1

        precision = true_detected_count/len(detected_object_list) if len(detected_object_list) != 0 else 1.0
        recall = true_detected_count/len(true_object_list) if len(true_object_list) != 0 else 1.0

        return Accuracy(precision, recall)

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

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--folder_path', type=str, help='enter a folder path without last /\nexample: ./sample/ILSVRC2015/Data/VID/snippets/val')

    args = parser.parse_args()

    accuracy_list = []
    for video_file_name in os.listdir(args.folder_path):
        video_folder_name = '.'.join(video_file_name.split('.')[:-1])
        print(f'calculating {video_folder_name}...')
        folder_path = f'./output/{video_folder_name}/detections'

        true_object_file = f'./output/{video_folder_name}/ground_truth/ground_truth.csv'
        true_object_list = DetectedObject.csv_to_detected_object_list(true_object_file)

        for detected_file in os.listdir(folder_path):
            detected_object_list = DetectedObject.csv_to_detected_object_list(f'{folder_path}/{detected_file}')
            cal = AccuracyCalculator(true_object_list, detected_object_list)
            accuracy = cal.run()
            accuracy_list.append(accuracy)

    calculator = ApCalculator(accuracy_list)
    print('==========================================')
    print('calculating...')
    calculator.run()
