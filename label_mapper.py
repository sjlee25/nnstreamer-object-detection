import json
from argparse import ArgumentParser
import os

class LabelMapper:
    def __init__(self, label_file_path):
        folder_name = '/'.join(label_file_path.split('/')[:-1])
        dirs = os.listdir(folder_name)
        for file_name in dirs:
            if file_name.split('.')[-1] == 'json':
                json_file_path = f'{folder_name}/{file_name}'

        self.labels = []

        with open(json_file_path) as json_file:
            json_data = json.load(json_file)
            self.data_set_label_dict = json_data["data_set_labels"]
        json_file.close()

        with open(label_file_path, "r", encoding='utf-8') as label_file:
            for line in label_file.readlines():
                if line[-1] == '\n':
                    line = line[:-1]
                self.labels.append(line)
        label_file.close()
    
    def get_label(self, index):
        return self.labels[index]
    
    def get_data_set_label(self, index):
        return self.data_set_label_dict[self.labels[index]]        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--label_file', type=str, help='path of label file')

    args = parser.parse_args()

    label_file = args.label_file

    label_mapper = LabelMapper(label_file)
    print(label_mapper.get_label(2))
    print(label_mapper.get_data_set_label(2))