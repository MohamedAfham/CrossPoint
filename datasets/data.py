import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import random
import math
from PIL import Image
from .plyfile import load_ply
from . import data_utils as d_utils
import torchvision.transforms as transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_DIR = 'data/ShapeNetRendering'

trans_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])
    
trans_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

def load_modelnet_data(partition):
    DATA_DIR = '/mnt/sdb/public/data/common-datasets'
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_ScanObjectNN(partition):
    BASE_DIR = '/mnt/sdb/public/data/common-datasets/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label

def load_shapenet_data():
    DATA_DIR = '/mnt/sdb/public/data/common-datasets'
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_filepath = []

    # print('-'*5, 'IN load_shapenet_data()')
    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        # print('-'*5, pcs)
        all_filepath += pcs
        
    # print('all_filepath', all_filepath)
    return all_filepath

def get_render_imgs(pcd_path):
    # print('-'*5, 'IN get_render_imgs()')

    # print('-'*5, pcd_path)
    path_lst = pcd_path.split('/')

    # path_lst[1] = 'ShapeNetRendering'
    path_lst[-3] = 'ShapeNetRendering'  # changed by jerry, because the directory of download datasets is different
    path_lst[-1] = path_lst[-1][:-4]

    path_lst.append('rendering')
    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))

    # print('-'*5, img_path_list)
    
    return img_path_list
        
class ShapeNetRender(Dataset):
    def __init__(self, img_transform = None, n_imgs = 1):
        self.data = load_shapenet_data()
        self.transform = img_transform
        self.n_imgs = n_imgs
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
        render_img_path = random.choice(get_render_imgs(pcd_path))
        # render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)
        # render_img_list = []
        # for render_img_path in render_img_path_list:
        render_img = Image.open(render_img_path).convert('RGB')
        render_img = self.transform(render_img)  #.permute(1, 2, 0)
            # render_img_list.append(render_img)
        pointcloud_1 = load_ply(self.data[item])
        # pointcloud_orig = pointcloud_1.copy()
        pointcloud_2 = load_ply(self.data[item])
        point_t1 = trans_1(pointcloud_1)
        point_t2 = trans_2(pointcloud_2)

        # pointcloud = (pointcloud_orig, point_t1, point_t2)
        pointcloud = (point_t1, point_t2)
        return pointcloud, render_img # render_img_list

    def __len__(self):
        return len(self.data)
    
class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
        
        

