import torch
import warnings
import wandb
import numpy as np
import random
import os
import sys

from omegaconf import OmegaConf


def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))

    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)

    torch.set_float32_matmul_precision('high')


def open_log(cfg):
    print(cfg)
    os.makedirs('logs/' + cfg.tag, exist_ok=True)
    if cfg.deploy:
        fname = 'logs/' + cfg.tag + '/' + cfg.run_name + ".log"
        fout = open(fname, "a", 1)
        sys.stdout = fout
        sys.stderr = fout
        print(cfg)
        print(wandb.run.id)
        return fout


def init_wandb(cfg, project_name):
    if cfg.deploy:
        wandb.init(project=project_name)
        wandb.run.name = wandb.run.id
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg))


def cleanup(cfg, fp):
    if cfg.deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()


def sanity_checks(cfg):
    # Check if BF16 is supported
    if not torch.cuda.is_available():
        warnings.warn("WARNING: running on CPU", UserWarning)
    else:
        if not torch.cuda.is_bf16_supported():
            warnings.warn("WARNING: running without BF16", UserWarning)

        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")
