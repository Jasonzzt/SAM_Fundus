import os
import numpy as np
import torch
from torch import nn
import warnings
from PIL import Image
import pandas as pd
from torchvision import transforms
import warnings
from collections import Counter
import cv2
from torch.utils.data import WeightedRandomSampler
warnings.filterwarnings("ignore")
data_transform = {
    'train': transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
        # transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        # transforms.CenterCrop(224),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, coordinate, category):
        self.data = data_root
        self.label = data_label
        self.coordinate = coordinate
        self.category = category

    def __getitem__(self, index):

        # data0 = Image.open(self.data[index])
        data0 = cv2.imread(self.data[index])
        data0 = cv2.cvtColor(data0, cv2.COLOR_BGR2RGB)
        # data0 = data_transform[self.category](data0)
        # data0 = np.array(data0)
        # data0 = torch.tensor(data0)
        labels = torch.from_numpy(np.array(self.label[index]))
        # coordinate = np.array(self.coordinate[index])
        coordinate = torch.tensor(self.coordinate[index])

        return data0, labels, coordinate
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼

    def __len__(self):
        return len(self.data)

class GetLoader2(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_od, coordinate, category):
        self.data = data_root
        self.od = data_od
        self.coordinate = coordinate
        self.category = category

    def __getitem__(self, index):

        # data_orig = Image.open(self.data[index]).convert('L')
        # data_orig = Image.open(self.data[index])
        data_orig = cv2.imread(self.data[index])
        data_od = Image.open(self.od[index])
        data_orig = cv2.cvtColor(data_orig, cv2.COLOR_BGR2RGB)
        # data_od = cv2.cvtColor(data_od, cv2.COLOR_BGR2RGB)
        # data_orig = data_transform[self.category](data_orig)
        # data_orig = np.array(data_orig)
        data_od = data_transform[self.category](data_od)
        # data_orig = torch.tensor(data_orig)
        # data_od = np.array(data_od)
        # data_od = torch.tensor(data_od)
        # coordinate = np.array(self.coordinate[index])
        coordinate = torch.tensor(self.coordinate[index])

        return data_orig, data_od, coordinate
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼

    def __len__(self):
        return len(self.data)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_data():
    train_label = []
    train_img = []
    train_coordinate = []
    data_train = pd.read_csv("../../dataAfterProcess/IDRiD/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
    data_coordinate_train = pd.read_csv("../../dataAfterProcess/IDRiD/C. Localization/C. Localization/2. Groundtruths/1. Optic Disc Center Location/a. IDRiD_OD_Center_Training Set_Markups.csv")
    data_train_array = np.array(data_train)
    data_coordinate_train_array = np.array(data_coordinate_train.T[:3].T[:413])
    for i in range(len(data_train)):
        train_img.append("../../dataAfterProcess/IDRiD/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set/" + str(data_train_array[i][0]) + ".jpg")
        train_label.append(data_train_array[i][1])
        train_coordinate.append(data_coordinate_train_array[i][1:])
    count = Counter(train_label)
    weights = []
    train_label = np.array(train_label).astype(int)
    for i in range(len(data_train)):
        weights.append(len(data_train)/count[train_label[i]])
    weights = torch.tensor(np.array(weights)).to(torch.float32).to(device)
    sampler = WeightedRandomSampler(weights,num_samples=len(data_train),replacement=True)
    train_img = np.array(train_img).astype(str)
    train_coordinate = np.array(train_coordinate).astype(int)

    valid_label = []
    valid_img = []
    valid_coordinate = []
    data_valid = pd.read_csv("../../dataAfterProcess/IDRiD/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")
    data_coordinate_valid = pd.read_csv("../../dataAfterProcess/IDRiD/C. Localization/C. Localization/2. Groundtruths/1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv")
    data_valid_array = np.array(data_valid)
    data_coordinate_valid_array = np.array(data_coordinate_valid.T[:3].T[:103])
    for i in range(len(data_valid)):
        valid_img.append("../../dataAfterProcess/IDRiD/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set/" + str(data_valid_array[i][0]) + ".jpg")
        valid_label.append(data_valid_array[i][1])
        valid_coordinate.append(data_coordinate_valid_array[i][1:])
    valid_label = np.array(valid_label).astype(int)
    valid_img = np.array(valid_img).astype(str)
    valid_coordinate = np.array(valid_coordinate).astype(int)
    train_data_IDRiD = torch.utils.data.DataLoader(
        GetLoader(train_img, train_label, train_coordinate, "train"), batch_size=2, sampler=sampler)
    valid_data_IDRiD = torch.utils.data.DataLoader(
        GetLoader(valid_img, valid_label, valid_coordinate, "valid"), batch_size=2, shuffle=True)
    
    return train_data_IDRiD, valid_data_IDRiD

def load_data2():
    train_img = []
    train_od = []
    train_coordinate = []
    data_coordinate_train = pd.read_csv("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/result_train.csv")
    data_coordinate_train_array = np.array(data_coordinate_train)
    i = 0
    for path in os.listdir("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set"):
        train_img.append("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set/" + path)
        train_coordinate.append(data_coordinate_train_array[i][1:])
        i += 1
    for path in os.listdir("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc"):
        train_od.append("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc/" + path)   
    train_img = np.array(train_img).astype(str)
    train_od = np.array(train_od).astype(str)
    train_coordinate = np.array(train_coordinate).astype(float)

    valid_img = []
    valid_od = []
    valid_coordinate = []
    data_coordinate_valid = pd.read_csv("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/result_test.csv")
    data_coordinate_valid_array = np.array(data_coordinate_valid)
    i = 0
    for path in os.listdir("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/1. Original Images/b. Testing Set"):
        valid_img.append("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/1. Original Images/b. Testing Set/" + path)
        valid_coordinate.append(data_coordinate_valid_array[i][1:])
        i += 1
    for path in os.listdir("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc"):
        valid_od.append("../../dataAfterProcess/IDRiD/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc/" + path)   
    valid_img = np.array(valid_img).astype(str)
    valid_od = np.array(valid_od).astype(str)
    valid_coordinate = np.array(valid_coordinate).astype(float)

    
    train_data_IDRiD_od = torch.utils.data.DataLoader(
        GetLoader2(train_img, train_od, train_coordinate, "train"), batch_size=2, shuffle=True)
    valid_data_IDRiD_od = torch.utils.data.DataLoader(
        GetLoader2(valid_img, valid_od, valid_coordinate, "valid"), batch_size=2, shuffle=True)

    return train_data_IDRiD_od, valid_data_IDRiD_od
    
