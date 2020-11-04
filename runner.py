from gt_position_extractor import GtPositionExtractor
from argparse import ArgumentParser
import pandas as pd
import os


parser = ArgumentParser()
parser.add_argument('--python_file_path', type=str, help='example: ./object_detection_tf.py')
parser.add_argument('--folder', type=str, help='enter folder path without last /\nexample: ./sample/ILSVRC2015/Data/VID/snippets/val')
parser.add_argument('--xml_file_folder', type=str, help='enter xml file folder path without last /\nexample: ./sample/ILSVRC2015/Annotations/VID')

args = parser.parse_args()
folder_path = args.folder
file_prefix = args.xml_file_folder
python_file = args.python_file_path
for file_name in os.listdir(folder_path):
    if file_name.split('.')[-1] == 'mp4':
        file_path = f'{folder_path}/{file_name}'
        for i in range(0,10):
            threshold_score = 0.1 + i*0.09
            os.system(f'python3 {python_file} --file {file_path} --threshold_score {threshold_score}')
        GtPositionExtractor('.'.join(file_name.split('.')[:-1]), file_prefix=file_prefix)

os.system(f'python3 real_ap_calculator.py --folder_path {folder_path}')