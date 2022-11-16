import logging
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
import os
from utils import *
from UNET import Unet
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(action)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class DifusionModel:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_scheldue().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_scheldue(self):
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_minus_aplha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        rand_value = torch.rand_like(x)
        return sqrt_alpha_hat * x + sqrt_minus_aplha_hat * rand_value, rand_value

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images ")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                tmp = (torch.ones(n) * i).long().to(self.device)
                predict_noise = model(x, tmp)
                k_aplha = self.alpha[tmp][:, None, None, None]
                k_aplha_hat = self.alpha_hat[tmp][:, None, None, None]
                beta = self.beta[tmp][:, None, None, None]
                if i > 1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(k_aplha) * (x - ((1 - k_aplha) / (torch.sqrt(1 - k_aplha_hat))) * predict_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1)/2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = Unet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = DifusionModel(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"C:\Users\Vadim\Desktop\folders\For myself\Python\datasets\cv\cifar10\train"# r"C:\Users\Vadim\Desktop\folders\For myself\Python\datasets\cv\cifar10\cifar-10-batches-py"
    args.device = "cpu"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()