"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import numpy as np

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * Variable(T.FloatTensor(x.size()).normal_(), requires_grad=False)
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(ImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h


class PatchImageDiscriminator(nn.Module):
    def __init__(self, n_channels, ndf=64, use_noise=False, noise_sigma=None):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h


class PatchVideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(PatchVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 4, 1, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
        )

    def forward(self, input):
        out = self.main(input).squeeze()
        return out


class VideoDiscriminator(nn.Module):
    def __init__(self, n_channels, n_output_neurons=1, bn_use_gamma=True, use_noise=False, noise_sigma=None, ndf=64):
        super(VideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv3d(ndf * 4, ndf * 8, 4, stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 8, n_output_neurons, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        out = self.main(input).squeeze()
        return out


class VideoGenerator(nn.Module):
    def __init__(self, n_channels, dim_z_content, dim_z_motion,
                 video_length, ngf=64):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.video_length = video_length

        dim_z = dim_z_motion + dim_z_content

        self.motion_dreamer = MotionDreamer(dim_z_motion)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_z, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def generate_videos(self, latent_content, latent_motion, video_len=16, autoregressive=False, motion_encoder=None):

        #  check if training or testing
        if autoregressive: # test
            
            # predict next latent motion
            generated_frames = []
            predicted_latent_motion, hidden = self.motion_dreamer(latent_motion.squeeze().unsqueeze(1), None)
            for frame_idx in range(video_len):
                
                # generate frames from latent motion and content
                latent_moco = torch.cat((latent_content.squeeze(), predicted_latent_motion.squeeze()), 1)
                predicted_frame = self.main(latent_moco.unsqueeze(-1).unsqueeze(-1)).unsqueeze(1)
                generated_frames.append(predicted_frame)

                # feed last frame back in (autoregressive)
                predicted_frame_cat = torch.cat(predicted_frame.split(1, 0), 1).squeeze(0)
                autoregressive_latent_motion = motion_encoder(predicted_frame_cat)

                predicted_latent_motion, hidden = self.motion_dreamer(autoregressive_latent_motion.squeeze().unsqueeze(1), hidden)
            
            # recover videos
            generated_videos = torch.cat(generated_frames, 1)
            return generated_videos

        else: # train

            # predict next latent motion
            predicted_latent_motion, _ = self.motion_dreamer(latent_motion)

            # build input for the CNN
            n_frames = predicted_latent_motion.size(1)
            latent_content = latent_content.unsqueeze(1).repeat(1, n_frames, 1) 
            latent_moco = torch.cat((latent_content, predicted_latent_motion), -1)
            latent_moco = torch.cat(latent_moco.split(1,0), 1).squeeze(0).unsqueeze(-1).unsqueeze(-1)

            # generate frames from latent motion and content
            generated_frames = self.main(latent_moco)

            # recover videos
            generated_videos = torch.stack(generated_frames.split(n_frames, 0))
            return generated_videos, predicted_latent_motion

    def generate_images(self, latent_content, latent_motion):
        latent_moco = torch.cat((latent_content, latent_motion), 1)
        return self.main(latent_moco)


class MoCoEncoder(nn.Module):
    def __init__(self, n_channels, moco_dims, ndf=64, use_noise=False, noise_sigma=None):
        super(MoCoEncoder, self).__init__()

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(n_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, moco_dims, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)


class MotionDreamer(nn.Module):
    def __init__(self, motion_dim, hidden_dim=256, dropout=0.0, depth=1):
        super(MotionDreamer, self).__init__()

        self.fc = nn.Linear(hidden_dim, motion_dim)
        self.gru = nn.GRU(motion_dim, hidden_dim, depth, batch_first=True, 
                          dropout=dropout * float(depth > 1))

    def forward(self, motion_latent_seq, hidden=None):
        out, hidden = self.gru(motion_latent_seq, hidden)
        return self.fc(out), hidden
    



# python src/train.py  \
#     --image_batch 4 \
#     --video_batch 2 \
#     --image_discriminator PatchImageDiscriminator \
#     --video_discriminator PatchVideoDiscriminator \
#     --print_every 1 \
#     --every_nth 2 \
#     data/actions logs/actions
