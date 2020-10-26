from typing import List
import pandas as pd
import os
from argparse import ArgumentParser
import xml.etree.ElementTree as ElementTree

class GtPositionExtractor:
    def __init__(self, file_name, train_folder_path = None):
        self.file_name = file_name
        self.train_folder_path = train_folder_path

    def run(self):
        file_prefix = "./sample/ILSVRC2015/Annotations/VID/"
        folder_path: str = file_prefix
        
        if('val' in self.file_name):
            folder_path = folder_path + 'val/' + self.file_name
        elif('train' in self.file_name):
            folder_path = folder_path + 'train/' + self.train_folder_path + '/' + self.file_name
        
        if not os.path.exists(f'./output/{self.file_name}'):
            os.makedirs(f'./output/{self.file_name}')
        self.csv_file_path = f"./output/{self.file_name}/ground_true.csv"
        file_number = 0
        self.result:List[List] = []
        try :
            while True:
                file_name = str(file_number).zfill(6)
                self.xml_parser(file_name, folder_path)
                file_number += 1
        except Exception as ex:
            print(ex)
            csv = pd.DataFrame(self.result)
            csv.to_csv(self.csv_file_path, header=False, index=False, mode = 'w')
            

    def xml_parser(self, file_name, folder_path):
        tree = ElementTree.parse(f'{folder_path}/{file_name}.xml')
        obj_list = tree.findall('object')
        for obj in obj_list:
            class_id = obj.find('name').text
            box = obj.find('bndbox')
            x_max = box.find('xmax').text
            x_min = box.find('xmin').text
            y_max = box.find('ymax').text
            y_min = box.find('ymin').text
            width = int(x_max) - int(x_min)
            height = int(y_max) - int(y_min)
            self.result.append([file_name, x_min, y_min, width, height, class_id, 0])

            


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--video_file_name', type=str, help='file name without extension')
    parser.add_argument('--train_folder_path', type=str, help="only required for 'train'")

    args = parser.parse_args()

    file_name = args.video_file_name
    train_folder_path = args.train_folder_path
    gtPositionExtractor = GtPositionExtractor(file_name, train_folder_path=train_folder_path)
    gtPositionExtractor.run()