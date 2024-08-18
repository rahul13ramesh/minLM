import hydra 
import torch
from utils.init import init_wandb, set_seed, open_log, cleanup
from utils.data import wikitext103_loader
from utils.nanogpt import nanoGPT
from utils.optimizer import configure_optimizers
from utils.runner import Runner


@hydra.main(config_path="./config", config_name="minconf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="icl")
    set_seed(cfg.seed)
    fp = open_log(cfg)

    trainloader = wikitext103_loader(cfg, train=True)
    testloader = wikitext103_loader(cfg, train=False)
    loaders = (trainloader, testloader)

    net = nanoGPT(cfg.net)
    opt = configure_optimizers(net, cfg.optimizer)

    runner = Runner(cfg, net, opt, loaders)
    runner.train()

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()



