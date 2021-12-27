# Decription
* The code is the official implementation of [QueryNet: Attack by Multi-Identity Surrogates](https://arxiv.org/abs/2105.15010)
* Authors: [Sizhe Chen](https://sizhechen.top), Zhehao Huang, Qinghua Tao, [Xiaolin Huang](http://www.pami.sjtu.edu.cn/en/xiaolin)
* This repository supports attack on
* MNIST    victim: resnet_preact,  densenet
* CIFAR10  victim: wrn-28-10-drop, gdas,       pyramidnet272
* ImageNet victim: inception_v3,   mnasnet1_0, resnext101_32x8d
* The experiments require 4 GPUs, each with 10G+ memory, and the CPU memory is better to be over 32G.


# Preparation
* Install dependencies

```
pip install -r requirements.txt
```

* To attack on CIFAR10: download [3 CIFAR10 victims](https://drive.google.com/u/0/uc?export=download&confirm=QBiP&id=1aXTmN2AyNLdZ8zOeyLzpVbRHZRZD0fW0) from Subspace Attack, and place them in `data/cifar10-models`
* To attack on ImageNet: download [ImageNet](https://imagenet.stanford.edu/) validation set and place them on `data/ILSVRC2012_img_val`


# Reproduce main results
* QueryNet attack on gdas (l_2 attack, eps=3):

```
python querynet.py --model=gdas --eps=3 --l2 --gpu=0,1,2,3 --num_srg=3 --use_nas --use_square_plus
```

* QueryNet attack on resnet_preact and densenet (l_infty attack, eps=0.3):

```
python querynet.py --model=resnet_preact,densenet --eps=76.5 --gpu=0,1,2,3 --num_srg=3 --use_nas --use_square_plus
```

* QueryNet attack on inception_v3 (l_2 attack, eps=5):

```
python querynet.py --model=inception_v3 --eps=5 --l2 --gpu=0,1,2,3 --num_srg=3 --use_nas --use_square_plus
```


# Reproduce ablation study
* Square baseline

```
python querynet.py --model=wrn-28-10-drop --eps=16 --gpu=0
```

* Square + S multiple identities

```
python querynet.py --model=wrn-28-10-drop --eps=16 --gpu=0,1,2,3 --num_srg=3
```

* Square + S multiple identities + Square_plus

```
python querynet.py --model=wrn-28-10-drop --eps=16 --gpu=0,1,2,3 --num_srg=3 --use_square_plue
```

* Square + S multiple identities + Square_plus + NAS ==> QueryNet

```
python querynet.py --model=wrn-28-10-drop --eps=16 --gpu=0,1,2,3 --num_srg=3 --use_square_plue --use_nas
```

* QueryNet: use 5 surrogates, repeat for 5 times

```
python querynet.py --model=wrn-28-10-drop --eps=16 --gpu=0,1,2,3,4,5 --num_srg=5 --use_square_plue --use_nas --run_times=5
```


# Files
```
├─attacker.py      # all crucial code in QueryNet.
├─querynet.py      # the main file to run, wrapping QueryNet at a high level.
├─surrogate.py     # surrogate model supporting NAS training.
├─victim.py        # includes victim models for different datasets.
├─utils.py         # helper functions, mostly the DataManager for Square+.
├─data             # pre-trained victims and is the data cache dir.
│  ├─cifar10-models
│  ├─ILSVRC2012_img_val
│  └─mnist-models  # we provide our well-trained MNIST victims directly in this folder.
├─logs             # experiment logs on attacking 4 MNIST victims and all ablation study.
│  ├─ablation
│  │  ├─l2-1
│  │  ├─l2-2
│  │  ├─l2-3
│  │  ├─l2-4
│  │  ├─l2-5
│  │  ├─linfty-12
│  │  ├─linfty-16
│  │  ├─linfty-20
│  │  ├─linfty-4
│  │  └─linfty-8
│  └─mnist
├─models           # 3rd party code from Subpace Attack.
├─PCDARTS          # 3rd party code, slightly changed for QueryNet's back-propagation.
├─pytorch_image_classification  # 3rd party code to support loading MNIST victims.
└─pretrained       # surrogates trained on ONLY query pairs, which is used as the cache to accelerate attacks. One could simply delete it to perform attacks from scratch, which leads to no performance drop but consumes more time only.
```

# 3rd party code
* [Subspace Attack](https://github.com/ZiangYan/subspace-attack.pytorch)
* [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS)
* [pytorch_image_classification](https://github.com/hysts/pytorch_image_classification)
