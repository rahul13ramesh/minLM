import torch
import tqdm
import time
import os
import wandb
import torch.nn.functional as F
from utils.optimizer import update_cosine_warmup_lr


class Runner:
    def __init__(self, cfg, net, opt, loaders):
        self.cfg = cfg
        self.loaders = loaders
        self.running_mfu = -1.0
        self.eval_time = 0.0
        self.optimizer = opt

        if cfg.net.compile:
            self.net = torch.compile(net)
        else:
            self.net = net

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
        eval_interval = self.cfg.log.eval_interval
        save_interval = self.cfg.log.save_interval

        # Initialize optimizer and net
        it = -1
        tr_loss = 0.0
        updates = 0
        epoch = 0.0
        optimizer.zero_grad(set_to_none=True)
        scaler = torch.GradScaler(dev, enabled=use_scaler)

        self.cur_time = time.time()

        net.train()
        net.to(dev)

        self.evaluate_model(0)

        while it < total_iters:
            print(f'Epoch {epoch}')
            epoch += 1

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

                # Train loss logging
                if it % log_interval == 0:
                    self.log_train_loss(it, tr_loss, lr)
                    tr_loss = 0.0
                else:
                    tr_loss += (loss.item() * grad_accumulation) / log_interval

                # Evaluate model perplexity
                if it % eval_interval == eval_interval - 1:
                    self.evaluate_model(it+1)

                if it % save_interval == 0:
                    self.save_model(it)

        # Evaluate model on entire validation set
        self.evaluate_model(it+1, full=True)
    
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

    def evaluate_model(self, it, full=False):
        start = time.time()
        net = self.net
        dev = self.cfg.device
        testloader = self.loaders[1]

        # Compute perplexity
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()

        # calculate average perplexity
        total_loss = 0.0
        total_tokens = 0.0
        with torch.inference_mode():
            for idx, dat in enumerate(testloader):
                dat = self.move_to_device(dat, dev)
                logits = net(dat[:, :-1])[:, -1]
                target = dat[:, -1]

                loss = criterion(logits, target)

                batch_size = dat.size(0)
                total_loss += torch.sum(loss) * batch_size
                total_tokens += batch_size

                if not full and idx >= self.cfg.log.eval_batches:
                    break

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(avg_loss).item()

        net.train()
        self.log_eval_perplexity(it, perplexity)

        self.eval_time += time.time() - start

        return perplexity

    def log_train_loss(self, it, loss, lr):
        if loss == 0.0:
            loss = float("inf")

        grad_accumulation = self.cfg.optimizer.grad_accumulation
        bs = self.cfg.data.bs

        dt = time.time() - self.cur_time
        dt -= self.eval_time

        dt = max(0, dt)
        self.eval_time = 0

        self.cur_time = time.time()
        mfu = self.net.estimate_mfu(bs * grad_accumulation, dt)

        if self.running_mfu < 0.0:
            self.running_mfu = mfu
        else:
            self.running_mfu = 0.9 * self.running_mfu + 0.1 * mfu

        print(f'Iter {it} | LR: {lr} | MFU: {self.running_mfu} | time {dt:.2f}s | Loss: {loss}')

        if self.cfg.deploy:
            wandb.log({
                'train_loss': loss, 'lr': lr, 'iter': it, 'mfu': self.running_mfu})

    def log_eval_perplexity(self, it, perplexity):
        print(f'Iter {it} | Perplexity: {perplexity}')

        if self.cfg.deploy:
            wandb.log({'eval_perplexity': perplexity, 'iter': it})

    def save_model(self, it):
        if self.cfg.deploy:
            fpath = os.path.join('./checkpoints/', self.cfg.tag, self.cfg.run_name)
            os.makedirs(fpath, exist_ok=True)
            torch.save(self.net.state_dict(), fpath + f'/model_{it}.pth')
