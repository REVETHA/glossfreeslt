# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, Sampler

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig

# *user-defined
import utils as utils
from dataloader.datasets import S2T_Dataset

# *basic
import os
import time
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import wandb
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger


# *metric
#from metrics import wer_list
from sacrebleu.metrics import BLEU, CHRF, TER

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import SoftTargetCrossEntropy
from timm.optim import AdamW

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image

from hpman.m import _
import hpargparse


# global definition
from definition import *

import torch.autograd.profiler as profiler
from models.SignCL import SignCL
cl_criterion = SignCL(max_distance=64.0)


class CheckpointableRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.indices = None

    def __iter__(self):
        if self.indices is None:
            self.indices = torch.randperm(len(self.data_source)).tolist()
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)

    def reset(self):
        self.indices = None

    def state_dict(self):
        return {'indices': self.indices}

    def load_state_dict(self, state):
        self.indices = state.get('indices')


def _checkpoint_state(model, optimizer, lr_scheduler, text_decoder,
                      optimizer_td, lr_scheduler_td, loss_scaler, epoch,
                      batch, min_loss, train_sampler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'text_decoder': text_decoder.state_dict(),
        'optimizer_td': optimizer_td.state_dict(),
        'lr_scheduler_td': lr_scheduler_td.state_dict(),
        'epoch': epoch,
        'batch': batch,
        'min_loss': min_loss,
        'train_sampler': train_sampler.state_dict(),
        'rng_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        },
    }
    if torch.cuda.is_available():
        state['rng_state']['cuda'] = torch.cuda.get_rng_state_all()
    if hasattr(loss_scaler, 'state_dict'):
        state['amp_scaler'] = loss_scaler.state_dict()
    return state


def _save_checkpoint(state, output_dir, filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / filename
    temporary_path = output_dir / f'.{filename}.tmp'
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def _restore_checkpoint(checkpoint, model, optimizer, lr_scheduler,
                        text_decoder, optimizer_td, lr_scheduler_td,
                        loss_scaler, train_sampler):
    model.load_state_dict(checkpoint['model'], strict=True)
    text_decoder.load_state_dict(checkpoint['text_decoder'], strict=True)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    if 'optimizer_td' in checkpoint:
        optimizer_td.load_state_dict(checkpoint['optimizer_td'])
    if 'lr_scheduler_td' in checkpoint:
        lr_scheduler_td.load_state_dict(checkpoint['lr_scheduler_td'])
    if 'amp_scaler' in checkpoint and hasattr(loss_scaler, 'load_state_dict'):
        loss_scaler.load_state_dict(checkpoint['amp_scaler'])
    rng_state = checkpoint.get('rng_state', {})
    if 'torch' in rng_state:
        torch.set_rng_state(rng_state['torch'])
    if 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])
    if 'python' in rng_state:
        random.setstate(rng_state['python'])
    if torch.cuda.is_available() and 'cuda' in rng_state:
        torch.cuda.set_rng_state_all(rng_state['cuda'])
    if 'train_sampler' in checkpoint:
        train_sampler.load_state_dict(checkpoint['train_sampler'])

def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) scripts', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-2, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.922, metavar='RATE',
                        help='LR decay rate for exponentialLR scheduler')
    parser.add_argument('--td_lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate for text decoder (default: 1e-3)')
    
     # * Baise params
    parser.add_argument('--output_dir', default='Checkpoints/Phoenix/GFSLT/vlp', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/phoenix/config1.yaml')

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)

    # * wandb params
    parser.add_argument("--log_all", action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    )
    parser.add_argument("--entity", type=str, default='user_name',
        help="wandb entity",
    )
    parser.add_argument("--wandb_name", type=str, default='VLP',
        help="wandb project",
    )

    # * Noise params
    parser.add_argument('--training-refurbish', default=True, type=bool)
    parser.add_argument('--noise-rate', default=0.15, type=float)
    parser.add_argument('--noise-type', default='omit_last', type=str, choices=['omit', 'omit_last'])
    parser.add_argument('--random-shuffle', default=False, type=bool)
    parser.add_argument('--loss-lambda', type=float, default=1.0, metavar='RATE',
                        help='lambda param')
    
    # others
    parser.add_argument('--accumulation_step',  type=int, default=1, help="accumulation step for loss backward.")
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='Maximum training batches per epoch; 0 means unlimited.')
    parser.add_argument('--max_eval_batches', type=int, default=0,
                        help='Maximum evaluation batches; 0 means unlimited.')
    parser.add_argument('--debug_only', action='store_true',
                        help='Stop after bounded training and validation without final evaluation.')
    parser.add_argument('--checkpoint-interval', type=int, default=250,
                        help='Save a mid-epoch checkpoint every N completed batches.')

    # Code Benchmark: Selection of Model Types
    parser.add_argument("--model_type", type=str, default='gfslt', help="options: gfslt, cico, signcl")

    # signcl specific parameters
    parser.add_argument('--zipf_factor', type=float, default=2.3, help='Zipf factor for signCL loss')


    return parser

def main(args, config):
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating tokenizer:")
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'])
    tokenizer.model_max_length = 300

    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer = tokenizer, config=config, args=args, phase='train', training_refurbish=True)
    print(train_data)
    train_sampler = CheckpointableRandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler,
                                 pin_memory=args.pin_mem,
                                 drop_last=True)

    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer = tokenizer, config=config, args=args, phase='dev', training_refurbish=True)
    print(dev_data)
    dev_sampler = torch.utils.data.RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 sampler=dev_sampler, 
                                 pin_memory=args.pin_mem)

    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer = tokenizer, config=config, args=args, phase='test', training_refurbish=True)
    print(test_data)
    test_sampler = torch.utils.data.RandomSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    from models.models import SLRCLIP, Text_Decoder
    model = SLRCLIP(config=config, model_type= args.model_type)
    model.to(device)
    print(model)


    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret =  model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    n_parameters = utils.count_parameters_in_MB(model)
    print(f'number of params: {n_parameters}M')


    optimizer = create_optimizer(args, model)
    if args.sched == "exp":
        lr_scheduler = scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=args.gamma,
            )
    else:
        lr_scheduler, _ = create_scheduler(args, optimizer)

    text_decoder = Text_Decoder(config).to(device)

    optimizer_td = AdamW(text_decoder.parameters(), lr=args.td_lr, weight_decay=0, betas=(0.9, 0.98))

    if args.sched == "exp":
        lr_scheduler_td = scheduler.ExponentialLR(
            optimizer=optimizer_td,
            gamma=args.gamma,
        )
    else:
        lr_scheduler_td = scheduler.CosineAnnealingLR(
                optimizer=optimizer_td,
                eta_min=1e-8,
                T_max=args.epochs,
            )

    TD_train_dict = dict(
        optimizer = optimizer_td,
        lr_scheduler = lr_scheduler_td,
        text_decoder = text_decoder
    )

    criterion = utils.KLLoss()
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    min_loss = np.inf
    resume_batch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        _restore_checkpoint(
            checkpoint, model, optimizer, lr_scheduler,
            TD_train_dict['text_decoder'], TD_train_dict['optimizer'],
            TD_train_dict['lr_scheduler'], loss_scaler, train_sampler,
        )
        min_loss = checkpoint.get('min_loss', np.inf)
        if not args.eval and 'epoch' in checkpoint:
            if 'batch' not in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                train_sampler.reset()
                resume_batch = 0
            else:
                resume_batch = checkpoint.get('batch', 0)
                if resume_batch >= len(train_dataloader):
                    args.start_epoch = checkpoint['epoch'] + 1
                    resume_batch = 0
                    train_sampler.reset()
                else:
                    args.start_epoch = checkpoint['epoch']

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, criterion, config, args.start_epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, config, args.start_epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if resume_batch == 0:
            train_sampler.reset()
        train_stats = train_one_epoch(
            args, model, criterion, train_dataloader, optimizer, device, epoch,
            config, PAD_IDX, loss_scaler, TD_train_dict, lr_scheduler, train_sampler,
            resume_batch, output_dir, min_loss,
        )
        resume_batch = 0
        lr_scheduler.step(epoch) 
        TD_train_dict['lr_scheduler'].step(epoch)
        # check min lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], args.min_lr)
        for param_group in TD_train_dict["optimizer"].param_groups:
            param_group['lr'] = max(param_group['lr'], args.min_lr/10)

        test_stats = evaluate(args, dev_dataloader, model, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)

        improved = min_loss > test_stats["loss"]
        if improved:
            min_loss = test_stats["loss"]

        if args.output_dir and not args.debug_only and utils.is_main_process():
            checkpoint_state = _checkpoint_state(
                model, optimizer, lr_scheduler, TD_train_dict['text_decoder'],
                TD_train_dict['optimizer'], TD_train_dict['lr_scheduler'],
                loss_scaler, epoch, len(train_dataloader), min_loss, train_sampler,
            )
            _save_checkpoint(checkpoint_state, output_dir, f'checkpoint_epoch_{epoch:04d}.pth')
            _save_checkpoint(checkpoint_state, output_dir, 'latest_checkpoint.pth')
            if improved:
                _save_checkpoint(checkpoint_state, output_dir, f'best_checkpoint_epoch_{epoch:04d}.pth')
                _save_checkpoint(checkpoint_state, output_dir, 'best_checkpoint.pth')
        
        print(f"* DEV loss {test_stats['loss']:.3f} Min DEV loss {min_loss}")
        if args.run:
            args.run.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'training/masked_lm_loss':train_stats['masked_lm_loss'], 'dev/dev_loss':test_stats['loss'], 'dev/min_loss': min_loss})


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.debug_only:
            print('DEBUG_ONLY_COMPLETE')
            return
        
        # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)

        dev_stats = evaluate(args, dev_dataloader, model, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, PAD_IDX, loss_scaler, TD_train_dict,
                    lr_scheduler, train_sampler, resume_batch, output_dir, min_loss, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 1
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)

    # with profiler.profile(use_cuda=True) as prof:
    for step, (src_input, tgt_input, masked_tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step < resume_batch:
            continue

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits, ground_truth, frames_feature = model(src_input, tgt_input)
            loss_text = criterion(logits["logits_per_text_image"], ground_truth)
            loss_image = criterion(logits["logits_per_image"], ground_truth)
            total_loss = (loss_text + loss_image) / 2.

            # signcl loss
            if args.model_type == "signcl":
                margin = max(10, int((frames_feature.shape[1] // tgt_input['input_ids'].shape[1] + 1) * args.zipf_factor)) * 2
                num_negative = 30
                margin = min(margin, int((frames_feature.shape[1] - num_negative) / 2))  # ensure num_frames margin for negative sampling
                cl_loss = cl_criterion(frames_feature, margin=margin)
                total_loss = total_loss + 0.01 * cl_loss

            total_loss = total_loss / args.accumulation_step # batch accumulation
        # batch accumulation
        if (step + 1) % args.accumulation_step == 0:
            loss_scaler(loss=total_loss, optimizer=optimizer, clip_grad=args.clip_grad, parameters=model.parameters())

        # update the text decoder parames
        if step % 5 == 0:
            TD_train_dict['optimizer'].zero_grad()
            with torch.cuda.amp.autocast():
                lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.model_txt)
                masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1)) * args.loss_lambda
                masked_lm_loss = masked_lm_loss / args.accumulation_step  # batch accumulation
            # batch accumulation
            if (step + 1) % args.accumulation_step == 0:
                loss_scaler(loss=masked_lm_loss, optimizer=TD_train_dict['optimizer'], clip_grad=args.clip_grad, parameters=TD_train_dict['text_decoder'].parameters())

        if args.debug_only and torch.cuda.is_available():
            print(
                f'DEBUG_BATCH {step + 1} '
                f'loss={total_loss.item():.6f} '
                f'masked_lm_loss={masked_lm_loss.item():.6f} '
                f'optimizer_step=SUCCESS '
                f'allocated_mb={torch.cuda.memory_allocated() / 1024**2:.2f} '
                f'reserved_mb={torch.cuda.memory_reserved() / 1024**2:.2f}',
                flush=True,
            )

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(masked_lm_loss=masked_lm_loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(td_lr=TD_train_dict['optimizer'].param_groups[0]["lr"])

        completed_batch = step + 1
        if (args.output_dir and utils.is_main_process()
                and args.checkpoint_interval > 0
                and completed_batch % args.checkpoint_interval == 0):
            checkpoint_state = _checkpoint_state(
                model, optimizer, lr_scheduler=lr_scheduler,
                text_decoder=TD_train_dict['text_decoder'],
                optimizer_td=TD_train_dict['optimizer'],
                lr_scheduler_td=TD_train_dict['lr_scheduler'],
                loss_scaler=loss_scaler, epoch=epoch, batch=completed_batch,
                min_loss=min_loss, train_sampler=train_sampler,
            )
            _save_checkpoint(
                checkpoint_state, output_dir,
                f'checkpoint_epoch_{epoch:04d}_batch_{completed_batch:06d}.pth',
            )
            _save_checkpoint(checkpoint_state, output_dir, 'latest_checkpoint.pth')

        if args.max_train_batches and step + 1 >= args.max_train_batches:
            print(f'Maximum training batches reached: {args.max_train_batches}')
            break

    if args.run:
        args.run.log({'epoch':epoch+1,'epoch/train_loss':loss_value, 'epoch/masked_lm_loss':masked_lm_loss.item()})
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 1
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)

    with torch.no_grad():
        for step, (src_input, tgt_input, masked_tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq, header)):
            logits, ground_truth, frames_feature = model(src_input, tgt_input)
            loss_text = criterion(logits["logits_per_text_image"], ground_truth)
            loss_image = criterion(logits["logits_per_image"], ground_truth)
            total_loss = (loss_text + loss_image) / 2.

            # signcl loss
            if args.model_type == "signcl":
                margin = max(10, int((frames_feature.shape[1] // tgt_input['input_ids'].shape[1] + 1) * args.zipf_factor)) * 2
                num_negative = 30
                margin = min(margin, int((frames_feature.shape[1] - num_negative) / 2))  # ensure num_frames margin for negative sampling
                cl_loss = cl_criterion(frames_feature, margin=margin)
                total_loss = total_loss + 0.01 * cl_loss

            lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.model_txt)
            masked_lm_loss = loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].cuda().view(-1))


            metric_logger.update(loss=total_loss.item())
            metric_logger.update(masked_lm_loss=masked_lm_loss.item())

            if args.debug_only and torch.cuda.is_available():
                print(
                    f'DEBUG_EVAL_BATCH {step + 1} '
                    f'loss={total_loss.item():.6f} '
                    f'masked_lm_loss={masked_lm_loss.item():.6f} '
                    f'allocated_mb={torch.cuda.memory_allocated() / 1024**2:.2f} '
                    f'reserved_mb={torch.cuda.memory_reserved() / 1024**2:.2f}',
                    flush=True,
                )

            if args.max_eval_batches and step + 1 >= args.max_eval_batches:
                print(f'Maximum evaluation batches reached: {args.max_eval_batches}')
                break


    if args.run:
        args.run.log({'epoch':epoch+1,'epoch/dev_loss':total_loss.item()})
    
    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def setup_run(args, config):
     # TODO: Please set the wand API key as an environment. Check if it was set.
    wandb_user_key = os.environ.get("WANDB_API_KEY", None)
    if wandb_user_key is None:
        print("Warning: WANDB_API_KEY not set, logging disabled")
        os.environ["WANDB_MODE"] = "disabled"

    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.wandb_name,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.wandb_name,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1]
        else:
            os.environ["WANDB_MODE"] = 'disabled'
            run = False

    return run
if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    config = utils.expand_env_vars(config)
    
    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)