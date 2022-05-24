from __future__ import print_function
import os
import datetime
import wandb
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# for distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from datasets.shapenet_part import ShapeNetPart
from models.dgcnn import DGCNN_partseg
from torch.utils.data import DataLoader
from util import AverageMeter, cal_loss, IOStream
from parser import args


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')

    # to fix BlockingIOError: [Errno 11]
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    os.environ["WANDB_BASE_URL"] = args.wb_url
    wandb.login(key=args.wb_key)


def setup(rank):
    # initialization for distibuted training on multiple GPUs
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)

    
def cleanup():
    dist.destroy_process_group()


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    if not visual:
        label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def train(rank):
    # initialize wandb once is enough
    if rank == 0:
        wandb.init(project="CrossPoint", name=args.exp_name)

    setup(rank)

    log_file = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
    io = IOStream('outputs/' + args.exp_name + f'/{log_file}', rank=rank)

    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points, class_choice=args.class_choice)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    
    """
    parameter `drop_last` in DataLoader():
        set to True to drop the last incomplete batch, 
        if the dataset size is not divisible by the batch size. 
        If False and the size of dataset is not divisible by the 
        batch size, then the last batch will be smaller. 
        (default: False)
    """
    samples_per_gpu = args.batch_size // args.world_size
    train_loader = DataLoader(
                    train_dataset, 
                    sampler=train_sampler,
                    batch_size=samples_per_gpu, 
                    shuffle=False, 
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False)

    val_dataset = ShapeNetPart(partition='test', num_points=args.num_points, class_choice=args.class_choice)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    val_samples_per_gpu = args.test_batch_size // args.world_size
    test_loader = DataLoader(val_dataset, 
                            sampler=val_sampler,
                            batch_size=val_samples_per_gpu, 
                            shuffle=False, 
                            num_workers=0, 
                            pin_memory=True,
                            drop_last=False)
    

    # in DGCNN and DGCNN_partseg, args.rank is used to specify the device where get_graph_feature() is executed
    args.rank = rank

    #Try to load models
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    model = DGCNN_partseg(args, seg_num_all, pretrain=False).to(rank)
    model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # load pretrained model in train_crosspoint.py
    map_location = torch.device('cuda:%d' % rank)
    pretrained_path = args.model_path
    model_ddp.load_state_dict(
        torch.load(pretrained_path, map_location=map_location), 
        strict=False)
    io.cprint("Model Loaded!!!")

    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(model_ddp.parameters(), lr=args.lr *100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(model_ddp.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        lr_scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        lr_scheduler = StepLR(opt, step_size=20, gamma=0.5)

    criterion = cal_loss
    
    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = AverageMeter()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []

        model_ddp.train()
        # required by DistributedSampler
        train_sampler.set_epoch(epoch)

        for data, label, seg in train_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(rank), label_one_hot.to(rank), seg.to(rank)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model_ddp(data, label_one_hot)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            train_loss.update(loss.item(), batch_size)
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
        if args.scheduler == 'cos':
            lr_scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                lr_scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)

        list0 = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(list0, train_loss.avg)
        train_loss_avg = np.mean(list0)

        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        list1 = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(list1, train_acc)
        train_acc_avg = np.mean(list1)

        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        list2 = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(list2, avg_per_class_acc)
        train_per_class_acc_avg = np.mean(list2)

        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)

        train_ious_on_single_gpu = np.mean(train_ious)
        list3 = [1.0 for _ in range(args.world_size)]
        dist.all_gather_object(list3, train_ious_on_single_gpu)
        train_ious_avg = np.mean(list3)

        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        outstr = '[%s] Train (%d/%d), loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % \
                 (time, epoch, args.epochs, train_loss_avg, train_acc_avg, train_per_class_acc_avg, train_ious_avg)
        io.cprint(outstr)

        ####################
        # Test
        ####################
        outstr, test_ious_avg, test_loss_avg, test_acc_avg, test_per_class_acc_avg = test(rank, epoch, model_ddp, test_loader, criterion)
        io.cprint(outstr)

        if rank == 0:
            wandb_log = {}
            wandb_log["Train Loss"] = train_loss_avg
            wandb_log["Train Accuracy"] = train_acc_avg
            wandb_log["Train Per Class Accuracy"] = train_per_class_acc_avg
            wandb_log["Train Mean IOU"] = train_ious_avg
            wandb_log["Test Loss"] = test_loss_avg
            wandb_log["Test Accuracy"] = test_acc_avg
            wandb_log["Test Per Class Accuracy"] = test_per_class_acc_avg
            wandb_log["Test Mean IOU"] = test_ious_avg
            wandb.log(wandb_log)

            if test_ious_avg >= best_test_iou:
                io.cprint('==> Saving Best Model ...')
                best_test_iou = test_ious_avg
                torch.save(model_ddp.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)

    # We should call wandb.finish() explicitly in multi processes training, 
    # otherwise wandb will hang in this process
    if rank == 0:
        wandb.finish()

    io.close()
    cleanup()


def test(rank, epoch, model, test_loader, criterion):
    # switch to eval() mode
    model = model.eval()

    seg_num_all = test_loader.dataset.seg_num_all
    seg_start_index = test_loader.dataset.seg_start_index

    test_loss = AverageMeter()
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in test_loader:
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(rank), label_one_hot.to(rank), seg.to(rank)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        seg_pred = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
        test_loss.update(loss.item(), batch_size)
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))

    list0 = [1.0 for _ in range(args.world_size)]
    dist.all_gather_object(list0, test_loss.avg)
    test_loss_avg = np.mean(list0)

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)

    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    list1 = [1.0 for _ in range(args.world_size)]
    dist.all_gather_object(list1, test_acc)
    test_acc_avg = np.mean(list1)

    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    list2 = [1.0 for _ in range(args.world_size)]
    dist.all_gather_object(list2, avg_per_class_acc)
    test_per_class_acc_avg = np.mean(list2)

    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)

    test_ious_on_single_gpu = np.mean(test_ious)
    list3 = [1.0 for _ in range(args.world_size)]
    dist.all_gather_object(list3, test_ious_on_single_gpu)
    test_ious_avg = np.mean(list3)

    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    outstr = '[%s] Test (%d/%d), loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % \
            (time, epoch, args.epochs, test_loss_avg, test_acc_avg, test_per_class_acc_avg, test_ious_avg)
    
    return outstr, test_ious_avg, test_loss_avg, test_acc_avg, test_per_class_acc_avg


if __name__ == "__main__":
    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log', rank=0)
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if args.cuda:
        num_devices = torch.cuda.device_count()
        if num_devices > 1:
            io.cprint('%d GPUs is available! Ready for DDP training' % num_devices)
            io.close()
            # Set seed for generating random numbers for all GPUs, and 
            # torch.cuda.manual_seed() is insufficient to get determinism for all GPUs
            torch.cuda.manual_seed_all(args.seed)
            mp.spawn(train, nprocs=args.world_size)
        else:
            io.cprint('Only one GPU is available, please use train_partseg.py')

    else:
        io.cprint('CUDA is unavailable! Exit')
        io.close()
