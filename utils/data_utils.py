import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, random_split

from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.data.data_utils import *
from torch.utils.data import Subset
from cords.utils.config_utils import load_config_data
import os.path as osp
# from cords.selectionstrategies.supervisedlearning import OMPGradMatchStrategy, RandomStrategy, CRAIGStrategy
from ray import tune

import imageio
from PIL import Image
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from cords.utils.data.data_utils import WeightedSubset

from tqdm.autonotebook import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""

def download_and_unzip(URL, root_dir):
  error_message = "Download is not yet implemented. Please, go to {URL} urself."
  raise NotImplementedError(error_message.format(URL))

def _add_channels(img, total_channels=3):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

"""Creates a paths datastructure for the tiny imagenet.

Args:
  root_dir: Where the data is located
  download: Download if the data is not there

Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:

"""
class TinyImageNetPaths:
  def __init__(self, root_dir, download=False):
    if download:
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                         root_dir)
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))

"""Datastructure for the tiny image dataset.

Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset

Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""
class TinyImageNetDataset(Dataset):
  def __init__(self, root_dir, mode='train', preload=True, load_transform=None,
               transform=None, download=False, max_samples=None):
    tinp = TinyImageNetPaths(root_dir, download)
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.preload = preload
    self.transform = transform
    self.transform_results = dict()

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]

    if self.preload:
      load_desc = "Preloading {} data...".format(mode)
      self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                               dtype=np.float32)
      self.label_data = np.zeros((self.samples_num,), dtype=np.int)
      for idx in tqdm(range(self.samples_num), desc=load_desc):
        s = self.samples[idx]
        img = Image.open(s[0]).convert("RGB")
        self.img_data[idx] = img
        if mode != 'test':
          self.label_data[idx] = s[self.label_idx]

      if load_transform:
        for lt in load_transform:
          result = lt(self.img_data, self.label_data)
          self.img_data, self.label_data = result[:2]
          if len(result) > 2:
            self.transform_results.update(result[2])

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    if self.preload:
      img = self.img_data[idx]
      lbl = None if self.mode == 'test' else self.label_data[idx]
    else:
      s = self.samples[idx]
   
      img = Image.open(s[0]).convert("RGB")
      lbl = None if self.mode == 'test' else s[self.label_idx]
    

    if self.transform:
      img = self.transform(img)
    
    
    return img, lbl

# Building custom dataset
class UltraMnistDataset(Dataset):
    def __init__(self, root_dir, X_train, y_train, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        label = self.y_train.iloc[index]
        image_path = f"{self.root_dir}/train/{self.X_train.iloc[index]}.jpeg"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label)

class DRDataset(Dataset):
    
    def __init__(self, root_dir, X_train, y_train, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, item):
        label = torch.tensor(self.y_train.iloc[item]).float()
        image_path = f"{self.root_dir}/train_images/{self.X_train.iloc[item]}.png"

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label

        


logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    
    if args.local_rank == 0:
        torch.distributed.barrier()

    # Loading the Dataset
    if args.dataset.name == "cifar10":
      trainset, valset, testset, num_cls = gen_dataset(datadir = "./data", dset_name= "cifar10", feature=None, args=args)

    elif args.dataset.name == "ultramnist":

      # Data transforms
      train_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])
      test_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])])


      #  Loading the train data
      train_csv_path = "./data/ultramnist512/train.csv"
      train_df = pd.read_csv(train_csv_path)
      # train_df = train_df.sample(10)
      train_df.head()

      # building training and validation sets
      X_train, X_valid, y_train, y_valid = train_test_split(train_df['id'], train_df['digit_sum'], test_size=0.1, random_state=42)
      print('Data lengths: ', len(X_train), len(X_valid), len(y_train), len(y_valid))

      # DataLoader
      trainset = UltraMnistDataset("./data/ultramnist512", X_train, y_train, train_transforms)
      # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
      testset = UltraMnistDataset("./data/ultramnist512",X_valid, y_valid,test_transforms)
    
      validation_set_fraction = 0.1
      num_fulltrn = len(trainset)
      num_val = int(num_fulltrn * validation_set_fraction)
      num_trn = num_fulltrn - num_val
      trainset, valset = random_split(trainset, [num_trn, num_val])

      num_cls = 28

    elif args.dataset.name == "aptos":
      # Data transforms
      train_transforms = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
      test_transforms = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

      #  Loading the train data
      train_csv_path = "./data/aptos/train.csv"
      train_df = pd.read_csv(train_csv_path)
      # train_df = train_df.sample(10)
      train_df.head()

      # building training and validation sets
      X_train, X_valid, y_train, y_valid = train_test_split(train_df['id_code'], train_df['diagnosis'], test_size=0.1, random_state=42)
      logger.info('Data lengths: ', len(X_train), len(X_valid), len(y_train), len(y_valid))

      # DataLoader
      trainset = DRDataset("./data/aptos", X_train, y_train, train_transforms)
      # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
      testset = DRDataset("./data/aptos",X_valid, y_valid,test_transforms)
      
      validation_set_fraction = 0.1
      num_fulltrn = len(trainset)
      num_val = int(num_fulltrn * validation_set_fraction)
      num_trn = num_fulltrn - num_val
      trainset, valset = random_split(trainset, [num_trn, num_val])

      num_cls = 1
      # 2966




    elif args.dataset.name == "imgnet":
    
    # else:

      if args.model_type == "Swin":
        train_transform = transforms.Compose([
          transforms.RandomResizedCrop((256, 256), scale=(0.05, 1.0)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])
          
        tst_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

      else:

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        tst_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

      trainset = TinyImageNetDataset(root_dir = "./data/tiny-imagenet-200", mode='train', preload=False, load_transform=None,transform=train_transform)

      testset = TinyImageNetDataset(root_dir = "./data/tiny-imagenet-200", mode='val', preload=False, load_transform=None,transform= tst_transform)

      validation_set_fraction = 0.1
      num_fulltrn = len(trainset)
      num_val = int(num_fulltrn * validation_set_fraction)
      num_trn = num_fulltrn - num_val
      trainset, valset = random_split(trainset, [num_trn, num_val])

      num_cls = 200

    trn_batch_size = args.train_batch_size
    val_batch_size = args.eval_batch_size
    tst_batch_size = args.eval_batch_size

    collate_fn = None
    batch_sampler = lambda _, __ : None
    
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    test_sampler = SequentialSampler(testset)
    
    drop_last = False
    
    if args.dss_args.type == "Full":
      wt_trainset = WeightedSubset(trainset, list(range(len(trainset))), [1] * len(trainset))

      dataloader = torch.utils.data.DataLoader(wt_trainset,
                                                  batch_size= trn_batch_size,
                                                  # shuffle= cfg.dataloader.shuffle,
                                                  pin_memory= True,
                                                  collate_fn= collate_fn,
                                                  num_workers= 4,
                                                  sampler= train_sampler)

      valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, sampler=val_sampler,
                                              shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last, num_workers=4)

      testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size, sampler=test_sampler,
                                                  shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last, num_workers=4)
    
    else:
      # Creating the Data Loaders
      trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size, sampler=train_sampler, shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last,num_workers=4)

      valloader = torch.utils.data.DataLoader(valset, batch_size=val_batch_size, sampler=val_sampler,
                                              shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last, num_workers=4)

      testloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size, sampler=test_sampler,
                                                  shuffle=False, pin_memory=True, collate_fn = collate_fn, drop_last=drop_last, num_workers=4)

    


    

    if args.dss_args.type == "Full":
      return dataloader, valloader, testloader , num_cls
    else:
      return trainloader, valloader, testloader, num_cls

