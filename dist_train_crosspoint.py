from __future__ import print_function
import os
import random
import argparse
import datetime
import torch
import math
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from torch.utils.data import DataLoader

# for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets.data import ShapeNetRender, ModelNet40SVM
from models.dgcnn import DGCNN, ResNet, DGCNN_partseg
from util import IOStream, AverageMeter
from parser import args


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

    WB_SERVER_URL = "http://202.112.113.239:28282"
    WB_KEY = "local-66548ed1d838753aa6c72555da8c798d184591b0"
    os.environ["WANDB_BASE_URL"] = WB_SERVER_URL
    wandb.login(key=WB_KEY)


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)

    
def cleanup():
    dist.destroy_process_group()


def train(rank):
    if rank == 0:
        wandb.init(project="CrossPoint", name=args.exp_name)

    setup(rank)
    # only write logs on the first gpu device whose rank=0
    io = IOStream('checkpoints/' + args.exp_name + '/run.log', rank=rank)
    
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_set = ShapeNetRender(transform, n_imgs = 2)
    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=rank)
    
    samples_per_gpu = args.batch_size // args.world_size
    train_loader = DataLoader(train_set, 
                              sampler=train_sampler,
                              batch_size=samples_per_gpu, 
                              shuffle=False, 
                              num_workers=0,
                              pin_memory=True
                              )

    # device = torch.device("cuda:%s" % args.gpu_id if args.cuda else "cpu")

    # in DGCNN and DGCNN_partseg, args.rank is used to specify the device where get_graph_feature() are executed
    args.rank = rank

    #Try to load models
    if args.model == 'dgcnn':
        # point_model = DGCNN(args).to(device)
        point_model = DGCNN(args).to(rank)
    elif args.model == 'dgcnn_seg':
        # point_model = DGCNN_partseg(args).to(device)
        point_model = DGCNN_partseg(args).to(rank)
    else:
        raise Exception("Not implemented")

    img_model = ResNet(resnet50(), feat_dim = 2048)
    img_model = img_model.to(rank)
        
    point_model_ddp = DDP(point_model, device_ids=[rank], find_unused_parameters=True)
    img_model_ddp = DDP(img_model, device_ids=[rank], find_unused_parameters=True)
        
    if args.resume:
        map_location = torch.device('cuda:%d' % rank)
        point_model_ddp.load_state_dict(
            torch.load(args.model_path, map_location=map_location)
        )
        img_model_ddp.load_state_dict(
            torch.load(args.img_model_path, map_location=map_location)
        )
        io.cprint("Model Loaded !!")
        
    # NOTE: 不同模型参数
    parameters = list(point_model_ddp.parameters()) + list(img_model_ddp.parameters())

    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(parameters, lr=args.lr, weight_decay=1e-6)

    lr_scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    criterion = NTXentLoss(temperature = 0.1).to(rank)
    
    best_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        if rank == 0:
            wandb_log = {}

        train_losses = AverageMeter()
        train_imid_losses = AverageMeter()
        train_cmid_losses = AverageMeter()
        
        # require by DistributedSampler
        train_sampler.set_epoch(epoch)

        point_model.train()
        img_model.train()
        io.cprint(f'Start training epoch: ({epoch}/{args.epochs})')
        for i, ((data_t1, data_t2), imgs) in enumerate(train_loader):
            data_t1, data_t2, imgs = data_t1.to(rank), data_t2.to(rank), imgs.to(rank)
            batch_size = data_t1.size()[0]
            
            opt.zero_grad()
            data = torch.cat((data_t1, data_t2))
            data = data.transpose(2, 1).contiguous()
            point_feats = point_model_ddp(data)[0]
            img_feats = img_model_ddp(imgs)
            
            point_t1_feats = point_feats[:batch_size, :]
            point_t2_feats = point_feats[batch_size: , :]
            
            loss_imid = criterion(point_t1_feats, point_t2_feats)        
            point_feats = torch.stack([point_t1_feats,point_t2_feats]).mean(dim=0)
            loss_cmid = criterion(point_feats, img_feats)
                
            total_loss = loss_imid + loss_cmid
            total_loss.backward()
            # for name, param in point_model_ddp.named_parameters():
            #     if param.grad is None:
            #         print('point', name)
            # for name, param in img_model_ddp.named_parameters():
            #     if param.grad is None:
            #         print('image', name)
            opt.step()
            
            train_losses.update(total_loss.item(), batch_size)
            train_imid_losses.update(loss_imid.item(), batch_size)
            train_cmid_losses.update(loss_cmid.item(), batch_size)
            
            if i % args.print_freq == 0:
                time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                outstr = '[%s] Epoch (%d), Batch(%d/%d), loss: %.6f, imid loss: %.6f, cmid loss: %.6f ' \
                        % (time, epoch, i, len(train_loader), train_losses.avg, train_imid_losses.avg, train_cmid_losses.avg)
                io.cprint(outstr)
        
        # In PyTorch 1.1.0 and later, you should call lr_scheduler.step() after optimizer.step()
        lr_scheduler.step()
    
        """ Explanation of the function dist.all_gather_object(list1, train_imid_losses.avg):
                list1: first parameter - a python list,
                        the length of list1 should be equavilent to the number of processes (world_size)

                train_imid_losses.avg: second parameter - a python object,
                        all_gather_object() gather the values of the second parameter across all devices within the process group, 
                        then broadcast these values into list1

                        e.g.:  if you have 6 GPUs, 
                                initialized list1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  
                                gathered values for train_imid_losses.avg through this function: 2.11, 4.14, 1.24, 4.48, 2.15, 3.13
                                after calling dist.all_gather_object(), list1 becomes [2.11, 4.14, 1.24, 4.48, 2.15, 3.13]
        """
        list0 = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(list0, train_losses.avg)
        train_loss_avg = np.mean(list0)

        list1 = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(list1, train_imid_losses.avg)
        train_imid_loss_avg = np.mean(list1)

        list2 = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(list2, train_cmid_losses.avg)
        train_cmid_loss_avg = np.mean(list2)

        outstr = 'Train %d, loss: %.6f, imid loss: %.6f, cmid loss: %.6f' % (epoch, train_loss_avg, train_imid_loss_avg, train_cmid_loss_avg)
        io.cprint(outstr)
        
        # Testing
        """
        这里SVM训练需要分布式吗？
            SVM是从 sklearn 导进来的，用不上 DDP 吧
        """
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=args.test_batch_size, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=args.test_batch_size, shuffle=True)

        feats_train = []
        labels_train = []
        point_model_ddp.eval()

        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(rank)
            with torch.no_grad():
                feats = point_model_ddp(data)[1]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        for i, (data, label) in enumerate(test_val_loader):
            labels = list(map(lambda x: x[0],label.numpy().tolist()))
            data = data.permute(0, 2, 1).to(rank)
            with torch.no_grad():
                feats = point_model_ddp(data)[1]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        
        io.cprint('Training SVM ...')
        model_tl = SVC(C = 0.1, kernel ='linear')
        model_tl.fit(feats_train, labels_train)

        io.cprint('Testing SVM ...')
        test_accuracy = model_tl.score(feats_test, labels_test)

        overall_accuracy = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(overall_accuracy, test_accuracy)
        test_accuracy_avg = np.mean(overall_accuracy)

        msg = f"Overall Linear Accuracy : {test_accuracy_avg}"
        io.cprint(msg)
        
        if rank == 0:
            wandb_log['Train Loss'] = train_loss_avg
            wandb_log['Train IMID Loss'] = train_imid_loss_avg
            wandb_log['Train CMID Loss'] = train_cmid_loss_avg
            wandb_log['Overall Linear Accuracy'] = test_accuracy_avg
            wandb.log(wandb_log)

        if test_accuracy_avg > best_acc:
            best_acc = test_accuracy_avg
            io.cprint('==> Saving Best Model...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'best_model.pth'.format(epoch=epoch))
            torch.save(point_model_ddp.state_dict(), save_file)
            
            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_model_best.pth')
            torch.save(img_model_ddp.state_dict(), save_img_model_file)
  
        if epoch % args.save_freq == 0:
            io.cprint('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model_ddp.state_dict(), save_file)
            save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_model_{epoch}.pth'.format(epoch=epoch))
            torch.save(img_model_ddp.state_dict(), save_img_model_file)
    
    io.cprint('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(point_model_ddp.state_dict(), save_file)
    save_img_model_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                         'img_model_last.pth')
    torch.save(img_model_ddp.state_dict(), save_img_model_file)

    # We should call wandb.finish() explicitly in multi processes training, otherwise wandb will hang in this process
    if rank == 0:
        wandb.finish()
    cleanup()
    io.close()


if __name__ == "__main__":
    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log', rank=0)
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    
    if args.cuda:
        io.cprint('CUDA is available! Using %d GPUs for DDP training' % args.world_size)
        io.close()

        torch.cuda.manual_seed(args.seed)
        mp.spawn(train, nprocs=args.world_size)

    else:
        io.cprint('CUDA is unavailable! Exit')
        io.close()
