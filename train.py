import hydra 

from utils.init import init_wandb, set_seed, open_log, cleanup




@hydra.main(config_path="./config/conf", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="icl")
    set_seed(cfg.seed)
    fp = open_log(cfg)

    # Get dataset

    cleanup(cfg, fp)



