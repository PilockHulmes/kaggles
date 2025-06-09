import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


import numpy as np
import os
from PIL import Image
import random

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()
        
        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        in_features = out_features
        
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, input_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        # A bunch of convolutions one after another
        model = [
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

class CycleGAN:
    def __init__(self, device, lr=0.0002, lambda_cycle=10, lambda_identity=0.5):
        self.device = device
        
        # Initialize generators and discriminators
        self.G = Generator().to(device)  # X -> Y
        self.F = Generator().to(device)  # Y -> X
        self.D_X = Discriminator().to(device)
        self.D_Y = Discriminator().to(device)
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            list(self.G.parameters()) + list(self.F.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        self.optimizer_D_X = optim.Adam(self.D_X.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_D_Y = optim.Adam(self.D_Y.parameters(), lr=lr, betas=(0.5, 0.999))
        
        

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
    
    def train(self, dataloader, epochs):
        self.lr_scheduler_G = optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(epochs, 0, epochs / 2).step)
        self.lr_scheduler_D_X = optim.lr_scheduler.LambdaLR(self.optimizer_D_X, lr_lambda=LambdaLR(epochs, 0, epochs / 2).step)
        self.lr_scheduler_D_Y = optim.lr_scheduler.LambdaLR(self.optimizer_D_Y, lr_lambda=LambdaLR(epochs, 0, epochs / 2).step)

        for epoch in range(epochs):

            for i, batch in enumerate(dataloader):
                # Set model input
                real_X = batch["X"].to(self.device)
                real_Y = batch["Y"].to(self.device)

                # Adversarial ground truths
                # valid = torch.ones((real_X.size(0), 1, 30, 30), requires_grad=False).to(self.device)
                valid = Variable(torch.Tensor(real_X.size(0), 1).fill_(1.0), requires_grad=False).to(self.device) # https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/train#L75
                # fake = torch.zeros((real_X.size(0), 1, 30, 30), requires_grad=False).to(self.device)
                fake = Variable(torch.Tensor(real_X.size(0), 1).fill_(0.0), requires_grad=False).to(self.device)
                
                # ------------------
                #  Train Generators
                # ------------------
                self.optimizer_G.zero_grad()
                
                # Identity loss
                loss_id_X = self.criterion_identity(self.F(real_X), real_X)
                loss_id_Y = self.criterion_identity(self.G(real_Y), real_Y)
                # loss_identity = (loss_id_X + loss_id_Y) / 2
                loss_identity = (loss_id_X + loss_id_Y) * 5 # https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/train#L107
                
                # GAN loss
                fake_Y = self.G(real_X)
                loss_GAN_G = self.criterion_GAN(self.D_Y(fake_Y), valid)
                fake_X = self.F(real_Y)
                loss_GAN_F = self.criterion_GAN(self.D_X(fake_X), valid)
                # loss_GAN = (loss_GAN_G + loss_GAN_F) / 2
                loss_GAN = loss_GAN_G + loss_GAN_F # https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/train#L115
                
                # Cycle loss
                recov_X = self.F(fake_Y)
                loss_cycle_X = self.criterion_cycle(recov_X, real_X)
                recov_Y = self.G(fake_X)
                loss_cycle_Y = self.criterion_cycle(recov_Y, real_Y)
                # loss_cycle = (loss_cycle_X + loss_cycle_Y) / 2
                loss_cycle = (loss_cycle_X + loss_cycle_Y) * 10 # https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/train#L123
                
                # Total loss
                loss_G = loss_GAN + self.lambda_cycle * loss_cycle + self.lambda_identity * loss_identity
                loss_G.backward()
                self.optimizer_G.step()
                
                # ---------------------
                #  Train Discriminator X
                # ---------------------
                self.optimizer_D_X.zero_grad()
                
                # Real loss
                loss_real = self.criterion_GAN(self.D_X(real_X), valid)
                # Fake loss
                loss_fake = self.criterion_GAN(self.D_X(fake_X.detach()), fake)
                # Total loss
                loss_D_X = (loss_real + loss_fake) * 0.5 # use * 0.5 to make sure the result is float
                loss_D_X.backward()
                self.optimizer_D_X.step()
                
                # ---------------------
                #  Train Discriminator Y
                # ---------------------
                self.optimizer_D_Y.zero_grad()
                
                # Real loss
                loss_real = self.criterion_GAN(self.D_Y(real_Y), valid)
                # Fake loss
                loss_fake = self.criterion_GAN(self.D_Y(fake_Y.detach()), fake)
                # Total loss
                loss_D_Y = (loss_real + loss_fake) * 0.5 # use * 0.5 to make sure the result is float
                loss_D_Y.backward()
                self.optimizer_D_Y.step()
                
                # Print progress
                if i % 5 == 0:
                    print(
                        f"[Epoch {epoch}/{epochs}] "
                        f"[Batch {i}/{len(dataloader)}] "
                        f"Loss D_X: {loss_D_X.item():.4f} "
                        f"Loss D_Y: {loss_D_Y.item():.4f} "
                        f"Loss G: {loss_G.item():.4f} "
                        f"Loss GAN: {loss_GAN.item():.4f} "
                        f"Loss cycle: {loss_cycle.item():.4f}"
                    )
                
                # Save images for visualization
                if i % 500 == 0:
                    self.save_images(epoch, i, real_X, real_Y, fake_X, fake_Y)
            self.lr_scheduler_D_X.step()
            self.lr_scheduler_D_Y.step()
            self.lr_scheduler_G.step()
    
    def save_images(self, epoch, batch, real_X, real_Y, fake_X, fake_Y):
        # Create directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        # Save generated images
        save_image(fake_X, f"images/{epoch}_{batch}_fake_X.png", normalize=True)
        save_image(fake_Y, f"images/{epoch}_{batch}_fake_Y.png", normalize=True)
        save_image(real_X, f"images/{epoch}_{batch}_real_X.png", normalize=True)
        save_image(real_Y, f"images/{epoch}_{batch}_real_Y.png", normalize=True)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_X, root_Y, transform=None, unaligned=False):
        self.transform = transforms.Compose(transform)
        self.unaligned = unaligned
        
        self.files_X = sorted([os.path.join(root_X, f) for f in os.listdir(root_X)])
        self.files_Y = sorted([os.path.join(root_Y, f) for f in os.listdir(root_Y)])
    
    def __getitem__(self, index):
        item_X = self.transform(Image.open(self.files_X[index % len(self.files_X)]))
        
        if self.unaligned:
            item_Y = self.transform(Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)]))
        else:
            item_Y = self.transform(Image.open(self.files_Y[index % len(self.files_Y)]))
        
        return {"X": item_X, "Y": item_Y}
    
    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))

if __name__ == '__main__':

    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 200
    batch_size = 4
    image_size = 256

    # Transforms
    transform = [
        transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # Dataloader
    dataloader = DataLoader(
        ImageDataset("./gan_to_art/data/photo_jpg", "./gan_to_art/data/monet_jpg", transform=transform, unaligned=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    # Initialize
    cyclegan = CycleGAN(device)

    # Train
    cyclegan.train(dataloader, epochs)
    
    torch.save(cyclegan.state_dict(), "model_weights.pth")