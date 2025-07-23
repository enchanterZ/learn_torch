

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)  #打开文件夹

    def __getitem__(self ,idx):
        img_name =self.img_path[idx]
        img_item_path = os.path.join(self.path,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir='dataset/train'
ants_label_dir ='ants'
ants_dataset= MyData(root_dir,ants_label_dir)
ants_dataset[5]
img, label = ants_dataset[0] # 使用索引方式获取第0个样本，这是Dataset的标准用法
print(f"获取到的第一张图片尺寸: {img.size}, 标签是: {label}")

# 如果你想显示图片(在Jupyter Notebook等环境中)
img.show()


