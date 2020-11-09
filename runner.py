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
if file_prefix[-1] != '/': file_prefix += '/'
python_file = args.python_file_path

video_file_lists = os.listdir(folder_path)
num_videos = len(video_file_lists)
video_cnt = 0

for file_name in video_file_lists:
    extension_idx = file_name.rfind('.')
    video_name = file_name[:extension_idx]
    extension = file_name[extension_idx+1:]

    if extension == 'mp4':
        video_cnt += 1
        file_path = f'{folder_path}/{file_name}'
        print(f"\n[Info] Running {file_name} ({video_cnt}/{num_videos})")

        os.system(f'python3 {python_file} --file {file_path} --threshold_score 0.3')

        if not os.path.exists(f'./output/{video_name}/ground_truth/ground_truth.csv'):
            gt_extractor = GtPositionExtractor(video_name, file_prefix=file_prefix)
            gt_extractor.run()
            print(f"[Info] Generated ./output/{video_name}/ground_truth/ground_truth.csv")
        else:
            print(f"[Info] Ground truth file already exists")

print("\n[Info] Calculating mAP...")
os.system(f'python3 real_ap_calculator.py --folder_path {folder_path}')

print("==========================================")
print(f"[Info] Execution completed! ({video_cnt}/{num_videos})\n")