import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[bool, int, int]:
    """Initialize distributed training from environment variables.

    Returns:
        (is_distributed, local_rank, world_size)
    """
    if "RANK" not in os.environ:
        return False, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return True, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
