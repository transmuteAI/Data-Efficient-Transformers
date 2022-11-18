# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging 
import argparse
import os
import random
import numpy as np

import sys
import wandb
from time import time

from datetime import datetime

sys.path.append(".")
sys.path.append("..")

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from models.networks import MobileNetV3_Network
import models.resnetv2 as models 
from models.modeling import VisionTransformer, CONFIGS
from models.swintransformerv2 import swinv2_base_window12to16_192to256_22kft1k

from utils.data_utils import get_loader
from utils.dist_util import get_world_size

from cords.utils.data.dataloader.SL.adaptive import CRAIGDataLoader, GradMatchDataLoader, GLISTERDataLoader, RandomDataLoader
from cords.utils.config_utils import load_config_data




logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
         self.reset()

    def reset(self):
         self.val = 0
         self.avg = 0
         self.sum = 0
         self.count = 0

    def update(self, val, n=1):
         self.val = val
         self.sum += val * n
         self.count += n
         self.avg =  self.sum /  self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    # config = CONFIGS[args.model_type]
    if args.dataset.name == "cifar10":
        num_classes = 10
    elif args.dataset.name == "imgnet":
        num_classes = 200
    elif args.dataset.name == "ultramnist":
        num_classes = 28

    elif args.dataset.name == "aptos":
        num_classes = 5

    if args.model_type == "BiT-M-R50x1":
        model = models.KNOWN_MODELS[args.model_type](head_size=num_classes, zero_head=True)
    
        if args.pretrained:
            logger.info("Loading pretrained model from %s" % args.pretrained_dir)
            model.load_from(np.load(f"./checkpoint/{args.model_type}.npz"))

    elif args.model_type == "ViT-B_16":
        config = CONFIGS[args.model_type]
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
        print(model)
        
        if args.pretrained:
            logger.info("Loading pretrained model from %s" % args.pretrained_dir)
            model.load_from(np.load(args.pretrained_dir))

    elif args.model_type == "Swin":
        model = swinv2_base_window12to16_192to256_22kft1k(pretrained=True, num_classes=num_classes)
        print(model)

    else:
    
        model = eval(f"{args.model_type}_Network(num_classes={num_classes})")
        print(model)
    
    
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step, epoch, best_acc):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)


    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Epoch: %d" % epoch)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    logger.info("Best till Accuracy: %2.5f" % best_acc)

    wandb.log({"Accuracy": accuracy})

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(cfg, model):
    """ Train the model """
    if cfg.local_rank in [-1, 0]:
        os.makedirs(cfg.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", cfg.name))

    cfg.train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps

    # Prepare dataset
    train_loader, val_loader, test_loader, num_cls = get_loader(cfg)

    cfg.dss_args.model = model
    cfg.dss_args.loss = nn.CrossEntropyLoss(reduction='none')
    cfg.dss_args.num_classes = cfg.model.numclasses
    cfg.dss_args.num_epochs = cfg.train_args.num_epochs
    cfg.dss_args.device = cfg.train_args.device
    cfg.dss_args.collate_fn = None

    if cfg.dss_args.type == "CRAIG":

        dataloader = CRAIGDataLoader(train_loader, val_loader, cfg.dss_args, logger,
                                        batch_size=cfg.train_batch_size,
                                        shuffle=cfg.dataloader.shuffle,
                                        pin_memory=True,
                                        collate_fn = cfg.dss_args.collate_fn)
        print(f"Length of dataloader {len(dataloader)}")

    elif cfg.dss_args.type == "GradMatch":
        cfg.dss_args.eta = cfg.learning_rate
        
        dataloader = GradMatchDataLoader(train_loader, val_loader, cfg.dss_args, logger,
                                             batch_size= cfg.train_batch_size,
                                             shuffle= cfg.dataloader.shuffle,
                                             pin_memory= True,
                                             collate_fn =  cfg.dss_args.collate_fn)

    elif cfg.dss_args.type == "GLISTER":
        cfg.dss_args.eta = cfg.learning_rate
        
        dataloader = GLISTERDataLoader(train_loader, val_loader, cfg.dss_args, logger,
                                             batch_size= cfg.train_batch_size,
                                             shuffle= cfg.dataloader.shuffle,
                                             pin_memory= True,
                                             collate_fn =  cfg.dss_args.collate_fn)

    elif cfg.dss_args.type == "Random":
        
        dataloader = RandomDataLoader(train_loader, cfg.dss_args, logger,
                                             batch_size= cfg.train_batch_size,
                                             shuffle= cfg.dataloader.shuffle,
                                             pin_memory= True,
                                             collate_fn =  cfg.dss_args.collate_fn) 

    elif cfg.dss_args.type == "Full":
        dataloader = train_loader

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.learning_rate,
                                momentum=0.9,
                                weight_decay=cfg.weight_decay)
    t_total = cfg.num_steps
    if cfg.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=cfg.warmup_steps, t_total=t_total)

    if cfg.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=cfg.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if cfg.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", cfg.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                cfg.train_batch_size * cfg.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if cfg.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)

    wandb.watch(model)

    model.zero_grad()
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    select_time = []
    iter_time = []
    for i in range(cfg.epochs):

        logger.info(f"Current Epoch {i}")

        model.train()
        
        epoch_iterator = tqdm(dataloader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=cfg.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(cfg.device) for t in batch)
            x, y, weights = batch
            
            if (step == 0) or (global_step % 10 == 0):
                logger.info("Starting iteration time recording")
                before_time = time()
            
            loss = model(x, y)
            loss = torch.dot(loss, weights / (weights.sum()))

            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            if cfg.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                losses.update(loss.item()*cfg.gradient_accumulation_steps)
                if cfg.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), cfg.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    final_time = time() - before_time
                    logger.info(f"time taken for 10 iterations {final_time}")
                    iter_time.append(final_time)

                    

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                logger.info("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val))

                if cfg.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)

                    wandb.log({"train/loss": losses.val})
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                    wandb.log({"train/lr": scheduler.get_lr()[0]})

                if (global_step % cfg.eval_every == 0) and cfg.local_rank in [-1, 0]:
                    accuracy = valid(cfg, model, writer, test_loader, global_step, i, best_acc)
                    if best_acc < accuracy:
                        save_model(cfg, model)
                        best_acc = accuracy
                    model.train()

                
        losses.reset()
       

    if cfg.local_rank in [-1, 0]:
        writer.close()
    logger.info(f"Global step: {global_step}")
    logger.info("Best Accuracy: {\t%f}" % best_acc)
    print("Best Accuracy: \t%f" % best_acc)
    logger.info(f"Iter time: {iter_time}")
    print(f"Iter time: {iter_time}")
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="./configs/craig_img_mobv3.py", type=str,
                        help="cfg file location")

    parser.add_argument("--name", default="craig_img_mobv3_pretrained", type=str,
                        help="cfg file location")
    
    parser.add_argument("--pretrained_dir", default="./checkpoint/ViT-B_16.npz", type=str,
                        help="cfg file location")

    parser.add_argument("--logger_dir", default="logger_out18", type=str,
                        help="cfg file location")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="gradient accumulation steps")

    parser.add_argument("--fraction", default=0.1, type=float,
                        help="fraction")

    parser.add_argument("--select_every", default=20, type=int,
                        help="select_every")
                        

    parser.add_argument("--num_steps", default=10000, type=int,
                        help="numsteps")

    parser.add_argument("--warmup_steps", default=500, type=int)
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--train_batch_size", default=512, type=int)
    parser.add_argument("--learning_rate", default=3e-2, type=float)
    
    parser.add_argument('--pretrained', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    
    
    args = parser.parse_args()

    cfg = load_config_data(args.cfg)



    # Setup CUDA, GPU & distributed training
    if cfg.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = torch.device("cuda", cfg.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        cfg.n_gpu = 1
    cfg.device = device

    cfg.name = "_".join([str(args.fraction), str(args.select_every), args.name, datetime.now().strftime("%b-%d_%H:%M:%S")])

    cfg.pretrained_dir = args.pretrained_dir
    cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    cfg.dss_args.fraction = args.fraction
    cfg.dss_args.select_every = args.select_every
    cfg.num_steps = args.num_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.epochs = 105
    cfg.pretrained = args.pretrained
    cfg.eval_every = args.eval_every
    cfg.logger_dir = args.logger_dir
    cfg.train_batch_size = args.train_batch_size
    cfg.learning_rate = args.learning_rate

    wandb.init(project="VIT 10k steps rand 21k cifar10", name=cfg.name, reinit=True, mode="disabled")
    wandb.config.update(cfg)

    # Setup logging
    os.makedirs(cfg.logger_dir, exist_ok=True)
    logging.basicConfig(filename=f"./{cfg.logger_dir}/{cfg.name}.log",
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if cfg.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16))

    # Set seed
    set_seed(cfg)

    # Model & Tokenizer Setup
    cfg, model = setup(cfg)

    # Training
    train(cfg, model)


if __name__ == "__main__":
    main()
