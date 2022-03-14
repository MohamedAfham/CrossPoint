# CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding (CVPR'22)
#### [Paper Link](https://arxiv.org/abs/2203.00680) | [Project Page](https://mohamedafham.github.io/CrossPoint/) 

> #### Abstract :
> Manual annotation of large-scale point cloud dataset for varying tasks such as 3D object classification, segmentation and detection is often laborious owing to the irregular structure of point clouds. Self-supervised learning, which operates without any human labeling, is a promising approach to address this issue. We observe in the real world that humans are capable of mapping the visual concepts learnt from 2D images to understand the 3D world. Encouraged by this insight, we propose CrossPoint, a simple cross-modal contrastive learning approach to learn transferable 3D point cloud representations. It enables a 3D-2D correspondence of objects by maximizing agreement between point clouds and the corresponding rendered 2D image in the invariant space, while encouraging invariance to transformations in the point cloud modality. Our joint training objective combines the feature correspondences within and across modalities, thus ensembles a rich learning signal from both 3D point cloud and 2D image modalities in a self-supervised fashion. Experimental results show that our approach outperforms the previous unsupervised learning methods on a diverse range of downstream tasks including 3D object classification and segmentation. Further, the ablation studies validate the potency of our approach for a better point cloud understanding.

## Citation

If you find our work, this repository, or pretrained models useful, please consider giving a star ⭐ and citation.
```bibtex
@inproceedings{afham2022crosspoint,
    title={CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding}, 
    author={Mohamed Afham and Isuru Dissanayake and Dinithi Dissanayake and Amaya Dharmasiri and Kanchana Thilakarathna and Ranga Rodrigo},
    booktitle={IEEE/CVF International Conference on Computer Vision and Pattern Recognition},
    month = {June},
    year={2022}
  }
```

## Dependencies

Refer `requirements.txt` for the required packages.

## Pretrained Models

CrossPoint pretrained models with DGCNN feature extractor are available [here.](https://drive.google.com/drive/folders/10TVEIRUBCh3OPulKI4i2whYAcKVdSURn?usp=sharing)

## Download data

Datasets are available [here](https://drive.google.com/drive/folders/1dAH9R3XDV0z69Bz6lBaftmJJyuckbPmR?usp=sharing). Run the command below to download all the datasets (ShapeNetRender, ModelNet40, ScanObjectNN, ShapeNetPart) to reproduce the results.

```
cd data
source download_data.sh
```

## Train CrossPoint

Refer `scripts/script.sh` for the commands to train CrossPoint.

## Downstream Tasks

### 1. 3D Object Classification 

Run `eval_ssl.ipynb` notebook to perform linear SVM object classification in both ModelNet40 and ScanObjectNN datasets.


### 2. Few-Shot Object Classification

Refer `scripts/fsl_script.sh` to perform few-shot object classification.

### 3. 3D Object Part Segmentation

Refer `scripts/script.sh` for fine-tuning experiment for part segmentation in ShapeNetPart dataset.

## Acknowledgements
Our code borrows heavily from [DGCNN](https://github.com/WangYueFt/dgcnn) repository. We thank the authors of DGCNN for releasing their code. If you use our model, please consider citing them as well.
