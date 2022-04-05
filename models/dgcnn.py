import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torch.nn.init as init

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, gpu_device=0):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda:%s' % gpu_device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

class DGCNN(nn.Module):
    def __init__(self, args, cls = -1):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.gpu_id = args.gpu_id
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        """
        self.conv1中，输入channel维度为6，这不是随便设的，而是 3*2 ，3代表原始输入点有三维坐标，经过get_graph_feature()
            拼接了邻居点的3维特征，成了6维

        """
        self.conv1 = nn.Sequential(nn.Conv2d(3*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        """
         不懂每个sequential内部的卷积，输入维度乘2是干什么？
            关键是前向传播，没有升维的操作呀，终于搞清楚了！
                是因为呀，get_graph_feature()函数的返回值，是 把每个点的特征，和k个近邻点的特征拼接了一下，所以特征维度变为原来2倍
                所以前向传播能理顺了
        """
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        
        if cls != -1:
            self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
            self.bn6 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=args.dropout)
            self.linear2 = nn.Linear(512, 256)
            self.bn7 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=args.dropout)
            self.linear3 = nn.Linear(256, cls)
        
        self.cls = cls
        
        self.inv_head = nn.Sequential(
                            nn.Linear(args.emb_dims * 2, args.emb_dims),
                            nn.BatchNorm1d(args.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.emb_dims, 256)
                            )

    def forward(self, x):
        batch_size = x.size(0)  # input x: (batch_size, feature_dim=3, num_points)
        x = get_graph_feature(x, k=self.k, gpu_device=self.gpu_id) # output x: (batch_size, feature_dim=6, num_points, k)
        x = self.conv1(x)   # output x: (batch_size, feature_dim=64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # output x: (batch_size, feature_dim=64, num_points)

        x = get_graph_feature(x1, k=self.k, gpu_device=self.gpu_id)     # output x: (batch_size, feature_dim=64*2, num_points, k)
        x = self.conv2(x)   # output x: (batch_size, feature_dim=64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # output x: (batch_size, feature_dim=64, num_points)

        x = get_graph_feature(x2, k=self.k, gpu_device=self.gpu_id)     # output x: (batch_size, feature_dim=64*2, num_points, k)
        x = self.conv3(x)   # output x: (batch_size, feature_dim=128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # output x: (batch_size, feature_dim=128, num_points)

        x = get_graph_feature(x3, k=self.k, gpu_device=self.gpu_id)     # output x: (batch_size, feature_dim=128*2, num_points, k)
        x = self.conv4(x)   # output x: (batch_size, feature_dim=256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # output x: (batch_size, feature_dim=256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # 每个点特征维度拼接共512维 = 64 + 64 + 128 + 256

        x = self.conv5(x)   # output x: (batch_size, feature_dim=args.emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)   # output x1: (batch_size, feature_dim=args.emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)   # output x2: (batch_size, feature_dim=args.emb_dims)
        x = torch.cat((x1, x2), 1)  # output x: (batch_size, feature_dim=args.emb_dims*2)
        
        feat = x    # --- output feat ---: (batch_size, feature_dim=args.emb_dims*2)
        if self.cls != -1:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)     # output x: (batch_size, feature_dim=512)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)     # output x: (batch_size, feature_dim=256)
            x = self.dp2(x)
            x = self.linear3(x)     # --- output x ---: (batch_size, num_classes)
        
        inv_feat = self.inv_head(feat)      # --- output feat ---: (batch_size, feature_dim=256)
        
        return x, inv_feat, feat 
    

class ResNet(nn.Module):
    def __init__(self, model, feat_dim = 2048):
        super(ResNet, self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Identity()
        
        self.inv_head = nn.Sequential(
                            nn.Linear(feat_dim, 512, bias = False),
                            nn.BatchNorm1d(512),
                            nn.ReLU(inplace=True),
                            nn.Linear(512, 256, bias = False)
                            ) 
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.inv_head(x)
        
        return x
    
    
class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x
    
    
class DGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all=None, pretrain=True):
    # def __init__(self, args):
        super(DGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.gpu_id = args.gpu_id
        self.pretrain = pretrain
        self.transform_net = Transform_Net(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.inv_head = nn.Sequential(
                            nn.Linear(args.emb_dims, args.emb_dims),
                            nn.BatchNorm1d(args.emb_dims),
                            nn.ReLU(inplace=True),
                            nn.Linear(args.emb_dims, 256)
                            )
        
        if not self.pretrain:
            self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                       self.bn7,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                       self.bn8,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp1 = nn.Dropout(p=args.dropout)
            self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                       self.bn9,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.dp2 = nn.Dropout(p=args.dropout)
            self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                       self.bn10,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l = None):
        batch_size = x.size(0)      # input x: (batch_size, feature_dim=3, num_points)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k, gpu_device=self.gpu_id)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k, gpu_device=self.gpu_id)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, gpu_device=self.gpu_id)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, gpu_device=self.gpu_id)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        
        if self.pretrain:
            # print("Pretrain")
            x = x.squeeze()
            inv_feat = self.inv_head(x)
            
            return x, inv_feat, x               # x: (batch_size, emb_dims), inv_feat: (batch_size, 256)
        
        else:   # 这块训练代码并没有写好加载数据
            l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
            l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

            x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
            x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

            x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

            x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
            x = self.dp1(x)
            x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
            x = self.dp2(x)
            x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
            x = self.conv11(x)                      # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
            
            return x
        