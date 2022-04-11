import argparse


# common settings for pretraining CrossPoint
parser = argparse.ArgumentParser(description='CrossPoint for Point Cloud Understanding')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                    help='Name of the experiment')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['dgcnn', 'dgcnn_seg'],
                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--eval', action='store_true',  help='evaluate the model')

parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                    help='Size of test batch)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of episode to train ')

parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')

parser.add_argument('--num_points', type=int, default=2048,
                    help='num of points to use')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')

parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='saved model path')
parser.add_argument('--img_model_path', type=str, default='', metavar='N',
                    help='saved image model path')

parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')

# training on single GPU device 
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu_id', type=int, default=0, help='specify the GPU device'
                    'to train of finetune model')

# distributed training on multiple GPUs
parser.add_argument('--rank', type=int, default=-1, help='the rank for current GPU or process, '
                    'ususally one process per GPU')
parser.add_argument('--backend', type=str, default='nccl', help='DDP communication backend')
parser.add_argument('--world_size', type=int, default=6, help='number of GPUs')
parser.add_argument('--master_addr', type=str, default='localhost', help='ip of master node')
parser.add_argument('--master_port', type=str, default='12355', help='port of master node')

# downstream task: Segmentation settings
parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                    choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'step'],
                    help='Scheduler to use, [cos, step]')


args = parser.parse_args()
