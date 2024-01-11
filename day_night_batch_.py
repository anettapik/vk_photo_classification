# CUDA_NUM = '0'
# DATA_DIR = 'north_current/img/1034_Архангельск'
# path = 'north_current/csv/1034_Архангельск.csv'
# weight = 'in_out_new/MiniNet2_run1.pt'

import argparse
import random
import os
import requests
import warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import sys

parser = argparse.ArgumentParser(description='Image Classification Script')
parser.add_argument('--cuda_num', type=str, help='CUDA device number')
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument('--data_dir', type=str, help='Path to the directory containing images')
parser.add_argument('--csv_path', type=str, help='Path to the CSV file')
parser.add_argument('--weight_path', type=str, help='Path to the pre-trained model weights')
args = parser.parse_args()

CUDA_NUM = args.cuda_num
DATA_DIR = args.data_dir
path = args.csv_path
weight = args.weight_path
BATCH_SIZE = args.batch_size

DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:'+CUDA_NUM)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# применения трансформации к картинке 
# resize до 224 * 256 
# делаем картинку тензорои и нармализуем
transform = transforms.Compose([
                           transforms.Resize(size=(224, 256)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])


# сконструировали модель
class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1028, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(1028)
        self.fc1 = nn.Linear(2056, 512)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.dropout1(F.relu(self.bn4(self.conv4(out))))
        out = F.max_pool2d(out, 4)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.max_pool2d(out, 4)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return out
    

# загрузили веса модели
model = MiniNet()
model = nn.DataParallel(model, device_ids=[int(CUDA_NUM)])
model.to(DEVICE)
model.module.load_state_dict(torch.load(weight, map_location=DEVICE))
model.eval()

file_name_list = []
totd = []
batch_size = BATCH_SIZE

# Итерация по батчам с использованием tqdm
file_list = os.listdir(DATA_DIR)
# in_out_new/data/csv/986_Северодвинск.csv
# sev_dv = pd.read_csv('in_out_new/data/csv/986_Северодвинск.csv')
# file_list = sev_dv['img_name'].to_list()
k = 0
# print(len(file_list)/batch_size)
print(len(file_list)/batch_size)

for batch_start in range(0, len(file_list), batch_size):
    batch_files = file_list[batch_start:batch_start + batch_size]
    batch_images = []
    batch_file_names = []

    for file_name in batch_files:
        img_path = os.path.join(DATA_DIR, file_name)
        img_to_classify = transform(Image.open(img_path)).unsqueeze(0)
        batch_images.append(img_to_classify)
        batch_file_names.append(img_path)

    # Преобразование батча в тензор
    batch_images = torch.cat(batch_images, dim=0)

    # Получение предсказаний для батча
    sig_layer = nn.Sigmoid()
    predictions = sig_layer(model(batch_images))


    # Обработка предсказаний для каждого изображения в батче
    for i in range(len(batch_files)):
        fx = 1 if predictions[i] > 0.5 else 0

        if fx == 0:
            totd.append('day')
        elif fx == 1:
            totd.append('night')
        else:
            totd.append('nan')

        file_name_list.append(batch_file_names[i])

    k = k+1
    print(k)
    
    # print(round((i+1)/(len(file_list)/batch_size)*100))



main_df = pd.read_csv(path)
intermediate_dictionary = {'images':file_name_list, 'time_of_the_day':totd}
df = pd.DataFrame(intermediate_dictionary)
# df['images'] = df['images'].apply(lambda x: x.replace('out/', '') if x.startswith('out/') else x)
df['images'] = df['images'].str.split('/').str[-1]
main_df =main_df.merge(df, left_on = 'img_name', right_on = 'images', how = 'left')
# main_df['url'] = main_df['url'].str[1:-1]
main_df.to_csv(f'{DATA_DIR}_dn.csv')

# ЗАПУСКАТЬ ЭТОТ КОД 
# python differents_code.your_script.py --cuda_num 0 --data_dir north_current/img/1034_Архангельск --csv_path north_current/csv/1034_Архангельск.csv --weight_path in_out_new/MiniNet2_run1.pt

# in_out_new/MiniNet2_run1.pt

# python3 day_night_batch_.py --cuda_num 0 --data_dir north_current/img/9901_Эгвекинот --csv_path north_current/csv/9901_Эгвекинот.csv --weight_path MiniNet2_run1.pt
# python3 day_night_batch_.py --cuda_num 0 --batch_size 8 --data_dir north_current/img/9901_Эгвекинот --csv_path north_current/csv/9901_Эгвекинот.csv --weight_path MiniNet2_run1.pt