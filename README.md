# Training-Vision-Transformers-with-only-2040-images
Official PyTorch implementation of training vision transformers with only 2040 images. 

Our paper is accepted to ECCV2022 and available at [[arxiv]](https://arxiv.org/abs/2201.10728).  
## Getting Started

### Prerequisites
* python 3
* PyTorch (= 1.6)
* torchvision (= 0.7)
* Numpy
* CUDA 10.1

### Install yq
```
sudo add-apt-repository ppa:rmescandon/yq 
sudo apt install yq
```

### Pre-training stage
- Pre-training stage using instance discrimination (c.f. run_deit_tiny_instance_discrimination_flowers.sh), run:
```
./scripts/run_deit_tiny_instance_discrimination_flowers.sh configs/run.yml
```

### Fine-tuning stage

- First, we fine-tune with 224x224 resolution, run:
- Change the input size to 224 inside the run.yml config
```
./scripts/run_deit_tiny_flowers.sh configs/run.yml
```

- Then, we continue to finetune with 448x448 resolution (c.f. run_deit_tiny_448_flowers.sh), run:
- Change the input size to 448 inside the run.yml config
```
./scripts/run_deit_tiny_flowers.sh  configs/run.yml
```

## Citation
Please consider citing our work in your publications if it helps your research.
```
@article{ViT2040,
   title         = {Training Vision Transformers with Only 2040 Images},
   author        = {Yun-Hao Cao, Hao Yu and Jianxin Wu},
   year          = {2022},
   journal = {The European Conference on Computer Vision}}
```
