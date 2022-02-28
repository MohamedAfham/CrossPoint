# Traning CrossPoint for classification
python train_crosspoint.py --model dgcnn --epochs 100 --lr 0.001 --exp_name crosspoint_dgcnn_cls --batch_size 20 --print_freq 200 --k 15


# Training CrossPoint for part-segmentation

python train_crosspoint.py --model dgcnn_seg --epochs 100 --lr 0.001 --exp_name crosspoint_dgcnn_seg --batch_size 20 --print_freq 200 --k 15


# Fine-tuning for part-segmentation

python train_partseg.py --exp_name dgcnn_partseg --pretrained_path dgcnn_partseg_best.pth --batch_size 8 --k 40 --test_batch_size 8 --epochs 300