import torch
import wandb
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

        optimizer = self.optimizer
        trainloader = self.loaders[0]

        # Get config
        dev = self.cfg.device
        total_iters = self.cfg.total_iters
        grad_accumulation = self.cfg.optimizer.grad_accumulation
        use_scaler = self.cfg.optimizer.use_scaler
        log_interval = self.cfg.log.log_interval

        # Initialize optimizer and net
        it = -1
        tr_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        scaler = torch.GradScaler(dev, enabled=use_scaler)

        net.train()
        net.to(dev)

        while it < total_iters:

            for dat in trainloader:
                if it >= total_iters:
                    break

                dat = self.move_to_device(dat, dev)

                # Update LR
                it, lr = update_cosine_warmup_lr(
                    it, self.cfg.optimizer, optimizer, total_iters)

                # Compute loss
                loss = self.compute_loss(dat)

                # Compute gradients
                scaler.scale(loss).backward()

                # Update model with gradients
                if it % grad_accumulation == grad_accumulation - 1:
                    self.update_model(scaler, optimizer)

                if it % log_interval  == 0:
                    self.log_train_loss(it, tr_loss, lr)
                    tr_loss = 0.0
                else:
                    tr_loss += (loss.item() * grad_accumulation) / log_interval

    
    def compute_loss(self, dat):
        net = self.net
        dev = self.cfg.device
        grad_accumulation = self.cfg.optimizer.grad_accumulation

        with torch.amp.autocast(device_type=dev,
                                dtype=torch.bfloat16):
            logits = net(dat[:, :-1])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                dat[:, 1:].flatten())
            loss = loss / grad_accumulation
        return loss

    def move_to_device(self, dat, dev):
        dat = dat['input_ids']
        dat = dat.to(dev)
        return dat

    def update_model(self, scaler, optimizer):
        net = self.net
        grad_clip = self.cfg.optimizer.grad_clip

        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    def log_train_loss(self, it, loss, lr):
        print(f'Iter {it} | Loss: {loss} | LR: {lr}')

        if self.cfg.deploy:
            wandb.log({'train_loss': loss, 'lr': lr, 'iter': it})
