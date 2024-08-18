import torch
import torch.nn.functional as F
from utils.optimizer import update_cosine_warmup_lr


class Runner:
    def __init__(self, cfg, net, opt, loaders):
        self.cfg = cfg
        self.net = net
        self.loaders = loaders
        self.optimizer = opt


    def train(self):
        net = self.net
        net.train()
        optimizer = self.optimizer
        trainloader = self.loaders[0]

        # Get config
        dev = self.cfg.device
        total_iters = self.cfg.total_iters

        grad_clip = self.cfg.optimizer.grad_clip
        use_scaler = self.cfg.optimizer.use_scaler
        grad_accumulation = self.cfg.optimizer.grad_accumulation

        # Initialize optimizer and net
        it = -1
        optimizer.zero_grad(set_to_none=True)
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

        while it < total_iters:

            for dat in trainloader:
                if it >= total_iters:
                    break

                dat = self.move_to_device(dat, dev)

                # Update LR
                it, lr = update_cosine_warmup_lr(
                    it, self.cfg.optimizer, optimizer, total_iters)

                # Compute loss
                with torch.amp.autocast(device_type=dev,
                                        dtype=torch.bfloat16):
                    logits = net(dat[:, :-1])
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        dat[:, 1:].flatten())
                    loss = loss / grad_accumulation

                # Update model
                scaler.scale(loss).backward()

                # Gradient clipping
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), grad_clip)

                if it % grad_accumulation == grad_accumulation - 1:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)


    def move_to_device(self, dat, dev):
        dat = dat['input_ids']
        dat = dat.to(dev)
        return dat

