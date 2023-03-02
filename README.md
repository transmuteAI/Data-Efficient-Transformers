# Data-Efficient Training of CNNs and Transformers with Coresets: A Stability Perspective

# Introduction

Coreset selection is among the most effective ways to reduce the training time of CNNs, however, only limited is known on how the resultant models will behave under variations of the coreset size, and choice of datasets and models. Moreover, given the recent paradigm shift towards transformer-based models, it is still an open question how coreset selection would impact their performance. There are several similar intriguing questions that need to be answered for a wide acceptance of coreset selection methods, and this paper attempts to answer some of these. We present a systematic benchmarking setup and perform a rigorous comparison of different coreset selection methods on CNNs and transformers. Our investigation reveals that under certain circumstances, random selection of subsets is more robust and stable when compared with the SOTA selection methods. We demonstrate that the conventional concept of uniform subset sampling across the various classes of the data is not the appropriate choice. Rather samples should be adaptively sampled based on the complexity of the data distribution for each class. Transformers are generally pretrained on large datasets, and we show that for certain target datasets, it helps to keep their performance stable at even very small coreset sizes. We further show that when no pretraining is done or when the pretrained transformer models are used with non-natural images (e.g. medical data), CNNs tend to generalize better than transformers at even very small coreset sizes. Lastly, we demonstrate that in the absence of the right pretraining, CNNs are better at learning the semantic coherence between spatially distant objects within an image, and these tend to outperform transformers at almost all choices of the coreset size.

# Installation

## Requirements

We have trained and tested our models on `Ubuntu 20.04`, `CUDA 11.3`, `GCC 9.3`, `Python 3.10` 

```
conda create -n cords -y
conda activate cords
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

# Datasets
We have used four datasets in our experiments. The details of the datasets are as follows: 
1. CIFAR-10
2. Tiny-ImageNet
3. UltraMNIST
4. APTOS-2019

Please download the respective dataset from the original sources and place them in the `data` folder.

# Training
For training, we have provided demo configs files in the `configs` folder. To train a model, run the following command with the appropriate config file and the name of the experiment. 
Example:

```
python train.py --cfg './configs/T-IMGNET/glis_img_vitb16.py' --name 'glis_img_vitb16_pretrained' --fraction 0.01 --num_steps 210 --warmup_steps 10 --select_every 20 --eval_every 2 --pretrained --logger_dir './vitb16_logger/IMGNET'
``` 

All the arguments are explained in the `train.py` file. Final logs will be saved in the `logger_dir` folder.

# Acknowledgements
We would like to thank the authors of the following repositories for their code and ideas: 
1. [big_transfer](https://github.com/google-research/big_transfer)
2. [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
3. [cords](https://github.com/decile-team/cords)

If you found Data-Efficent Transformers useful please consider citing these works as well.
