"""ReSwapper v2 training entry point.

Usage:
    # Single GPU
    python train_v2.py --config configs/default.yaml

    # Debug mode
    python train_v2.py --config configs/default.yaml --override configs/debug.yaml

    # Multi-GPU (3x RTX 6000 Ada or B200)
    torchrun --nproc_per_node=3 train_v2.py --config configs/default.yaml

    # Resume from checkpoint
    python train_v2.py --config configs/default.yaml --resume checkpoints/checkpoint-50000.pt

    # 1024px fine-tuning from 512px checkpoint
    python train_v2.py --config configs/default.yaml --override configs/finetune_1024.yaml \
        --resume checkpoints/checkpoint-500000.pt
"""

import argparse
import random

import numpy as np
import torch

from reswapper.config import ReSwapperConfig
from reswapper.training.trainer import Trainer
from reswapper.utils.distributed import setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReSwapper v2")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Override config files")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    all_configs = [args.config] + args.override
    config = ReSwapperConfig.from_yamls(all_configs)

    if args.resume:
        config.checkpoint.resume_from = args.resume
    if args.seed is not None:
        config.seed = args.seed

    # Setup distributed
    is_distributed, local_rank, world_size = setup_distributed()

    # Seed everything
    random.seed(config.seed + local_rank)
    np.random.seed(config.seed + local_rank)
    torch.manual_seed(config.seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + local_rank)

    # Create trainer and run
    trainer = Trainer(config, local_rank=local_rank, world_size=world_size)

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
