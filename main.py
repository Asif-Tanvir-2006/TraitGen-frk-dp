import os
import re
import json
import torch
import random
import logging
import argparse

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from dataset.dataset import CocoFormatDataset
from Model.model import TraitGen
from Model.VisionEncoder import VisionEncoder
from engine import *

def ddp_setup():
    """Initializes the distributed process group."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_args_parser():
    parser = argparse.ArgumentParser('XAI Training', add_help=False)

    parser.add_argument('--data_root', default='/home/paul/Paul/DATASETS/cub')
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--optimizer', default='adam', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size per GPU')
    parser.add_argument('--encoder_model', default="hf-hub:imageomics/bioclip")
    parser.add_argument('--encoder_op_dim', default=768, type=int)
    parser.add_argument('--decoder_model', default="openai-community/gpt2-medium")
    parser.add_argument('--streeing_prompt', default="species identification and corresponding textual explanation task.")
    parser.add_argument('--ann_dir', default='/kaggle/input/custom-ds', help='Path to custom JSON annotations')
    return parser

def main(args):
    # Setup DDP environment
    ddp_setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Only set up logging on the primary GPU process (rank 0)
    logger = None
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, "log.txt")
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s | %(message)s')
        logger = logging.getLogger(__name__)

    vision_encoder = VisionEncoder(args)
    preprocess = vision_encoder.preprocess

    train_dataset = CocoFormatDataset(args, ann_file=f'{args.ann_dir}/cub_train_split1.json', img_prefix=f'{args.data_root}/images', preprocess=preprocess)
    test_dataset = CocoFormatDataset(args, ann_file=f'{args.ann_dir}/cub_test_split1.json', img_prefix=f'{args.data_root}/images', preprocess=preprocess)

    # Use DistributedSampler to partition datasets across GPUs
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Sampler handles shuffling
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )

    # Instantiate model and wrap with DDP
    model = TraitGen(args, vision_encoder=vision_encoder).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(filter(
        lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        # Set epoch on sampler so shuffling varies each epoch
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_acc = validate(args, model, test_loader, device)

        # Log and save checkpoints only from rank 0
        if local_rank == 0:
            if logger:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            if epoch == args.epochs - 1:
                # Save model.module to get the underlying model weights (excluding DDP wrapper)
                save_checkpoint("best_model.pth", model.module, optimizer, epoch)

    destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('TraitGen Training', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
