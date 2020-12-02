from typing import List
import pandas as pd
import os
from argparse import ArgumentParser
import json

class GTObject:
    def __init__(self, line):
        data = line.split(',')
        self.frame = int(data[0])
        self.x = int(data[1])
        self.y = int(data[2])
        self.width = int(data[3])
        self.height = int(data[4])
        self.class_name = int(data[5])

class GtPositionExtractor:
    def __init__(self, file_name):
        self.file_name = file_name
        csv_folder_path = './output/'

        csv_folder_path += 'ground_truth'
        if not os.path.exists(csv_folder_path):
            os.makedirs(csv_folder_path)
        self.csv_file_path = csv_folder_path + file_name.split('/')[-1] + '/ground_truth.csv'

    def run(self):
        self.result:List[List] = []
        self.json_parser(self.file_name)
        csv = pd.DataFrame(self.result)
        csv.to_csv(self.csv_file_path, header=False, index=False, mode = 'w')

    def json_parser(self, file_name):
        with open(f"{file_name}.json") as json_file:
            json_data = json.load(json_file)
            annotation = json_data['annotation']
            for obj in annotation:
                bbox = obj['bbox']
                category_id = obj['category_id']
                image_id = obj['image_id']
                x_max = max(bbox[0], bbox[1])
                x_min = min(bbox[0], bbox[1])
                y_max = max(bbox[2], bbox[3])
                y_min = min(bbox[2], bbox[3])
                width = x_max - x_min
                height = y_max - y_min
                self.result.append([image_id, x_min, y_min, width, height, category_id, 0])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, help='json file path without file extension')

    args = parser.parse_args()

    file_name = args.file
    gtPositionExtractor = GtPositionExtractor(file_name)
    gtPositionExtractor.run()