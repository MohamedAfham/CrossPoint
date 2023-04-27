# CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding (CVPR'22)
#### [Paper Link](https://arxiv.org/abs/2203.00680) | [Project Page](https://mohamedafham.github.io/CrossPoint/) 

## Citation

If you find our work, this repository, or pretrained models useful, please consider giving a star ⭐ and citation.
```bibtex
@InProceedings{Afham_2022_CVPR,
    author    = {Afham, Mohamed and Dissanayake, Isuru and Dissanayake, Dinithi and Dharmasiri, Amaya and Thilakarathna, Kanchana and Rodrigo, Ranga},
    title     = {CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9902-9912}
}
```

# :rocket: News
* **(Mar 25, 2023)**
  * An implementation supporting PyTorchDistributedDataParallel (DDP) is available [here](https://github.com/auniquesun/CrossPoint-DDP). Thanks to [Jerry Sun](https://auniquesun.github.io/)
* **(Mar 2, 2022)**
  * Paper accepted at CVPR 2022 :tada: 
* **(Mar 2, 2022)** 
  * Training and evaluation codes for [CrossPoint](https://openaccess.thecvf.com/content/CVPR2022/html/Afham_CrossPoint_Self-Supervised_Cross-Modal_Contrastive_Learning_for_3D_Point_Cloud_Understanding_CVPR_2022_paper.html), along with pretrained models are released.

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
