"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import os
import time

import numpy as np

from logger import Logger

import torch
from torch import nn
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 2, 3, 1)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def videos_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(0, 1, 2, 3, 4)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')


def one_hot_to_class(tensor):
    a, b = np.nonzero(tensor)
    return np.unique(b).astype(np.int32)


class Trainer(object):
    def __init__(self, image_sampler, video_sampler, log_interval, train_batches, log_folder, use_cuda=False,
                 use_infogan=True):


        self.gan_criterion = nn.BCEWithLogitsLoss()

        self.image_sampler = image_sampler
        self.video_sampler = video_sampler

        self.video_batch_size = self.video_sampler.batch_size
        self.image_batch_size = self.image_sampler.batch_size

        self.log_interval = log_interval
        self.train_batches = train_batches

        self.log_folder = log_folder

        self.use_cuda = use_cuda
        self.use_infogan = use_infogan

        self.image_enumerator = None
        self.video_enumerator = None

    @staticmethod
    def ones_like(tensor, val=1.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    @staticmethod
    def zeros_like(tensor, val=0.):
        return Variable(T.FloatTensor(tensor.size()).fill_(val), requires_grad=False)

    def sample_real_image_batch(self):
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.image_sampler)

        batch_idx, batch = next(self.image_enumerator)
        b = batch
        if self.use_cuda:
            b = b.cuda()

        if batch_idx == len(self.image_sampler) - 1:
            self.image_enumerator = enumerate(self.image_sampler)

        return b

    def sample_real_video_batch(self):
        if self.video_enumerator is None:
            self.video_enumerator = enumerate(self.video_sampler)

        batch_idx, batch = next(self.video_enumerator)
        b = batch
        if self.use_cuda:
            b = b.cuda()

        if batch_idx == len(self.video_sampler) - 1:
            self.video_enumerator = enumerate(self.video_sampler)

        return b

    # def train_discriminator(self, discriminator, sample_true, generate_func, opt, batch_size, use_categories):
    #     opt.zero_grad()

    #     real_batch = sample_true()
    #     batch = Variable(real_batch['images'], requires_grad=False)

    #     # util.show_batch(batch.data)

    #     fake_batch, generated_categories = generate_func(batch_size)

    #     real_labels, real_categorical = discriminator(batch)
    #     fake_labels, fake_categorical = discriminator(fake_batch.detach())

    #     ones = self.ones_like(real_labels)
    #     zeros = self.zeros_like(fake_labels)

    #     l_discriminator = self.gan_criterion(real_labels, ones) + \
    #                       self.gan_criterion(fake_labels, zeros)

    #     if use_categories:
    #         # Ask the video discriminator to learn categories from training videos
    #         categories_gt = Variable(torch.squeeze(real_batch['categories'].long()), requires_grad=False)
    #         l_discriminator += self.category_criterion(real_categorical.squeeze(), categories_gt)

    #     l_discriminator.backward()
    #     opt.step()

    #     return l_discriminator

    def train_adversarial(self, generator,
                          image_discriminator, video_discriminator,
                          content_encoder, motion_encoder, 
                          opt_generator, opt_image_discriminator, opt_video_discriminator):


        ########################################################
        # train encoder, generator and discriminator on images #
        ########################################################

        # reset grads on image discriminator and generator
        opt_image_discriminator.zero_grad()

        # load image data
        real_image_batch = self.sample_real_image_batch()

        # extract latent content and motion
        latent_content = content_encoder(real_image_batch)
        latent_motion = motion_encoder(real_image_batch)

        # generate from latent content and motion
        generated_image_batch = generator.generate_images(latent_content, latent_motion)

        # get discriminator real / generated image labels
        real_image_labels = image_discriminator(real_image_batch)
        generated_image_labels = image_discriminator(generated_image_batch.detach())

        # build discriminator ground truth
        ones_image = self.ones_like(real_image_labels)
        zeros_image = self.zeros_like(generated_image_labels)

        # compute image discriminator loss
        l_image_discriminator = self.gan_criterion(real_image_labels, ones_image) + \
                                self.gan_criterion(generated_image_labels, zeros_image)
        
        # update discriminator, generator will be updated later
        l_image_discriminator.backward()
        opt_image_discriminator.step()


        ####################################
        # training discriminator on videos #
        ####################################
        
        # reset grads on video discriminator
        opt_video_discriminator.zero_grad()

        # load video data
        real_video_batch = self.sample_real_video_batch()
        
        # reshape and cat tensors to eval latents
        n_frames = real_video_batch.size(2)
        real_video_batch_perm = real_video_batch.permute(0, 2, 1, 3, 4)
        real_video_batch_cat = torch.cat(real_video_batch_perm.split(1, 0), 1).squeeze(0)

        # extract latent content and motion
        latent_content = content_encoder(real_video_batch_perm[:,0,:])
        latent_motion = motion_encoder(real_video_batch_cat)

        # recover previous order
        latent_content = torch.stack(latent_content.split(n_frames, 0)).squeeze()
        latent_motion = torch.stack(latent_motion.split(n_frames, 0)).squeeze()

        # prepare ground truth and input for video generator
        latent_motion_gt = latent_motion[:,1:,:] # ground truth is the shifted input
        latent_motion = latent_motion[:,:-1,:]

        # generate video
        generated_video_batch, predicted_motion_content = generator.generate_videos(latent_content, latent_motion)
        generated_video_batch = generated_video_batch.permute(0,2,1,3,4)

        # get discriminator real and generated video labels
        real_video_labels = video_discriminator(real_video_batch[:,:,1:,:])
        generated_video_labels = video_discriminator(generated_video_batch.detach())
        
        # build discriminator ground truth
        ones_video = self.ones_like(real_video_labels)
        zeros_video = self.zeros_like(generated_video_labels)

        # compute video discriminator loss
        l_video_discriminator = self.gan_criterion(real_video_labels, ones_video) + \
                                self.gan_criterion(generated_video_labels, zeros_video)

        # update video discriminator
        l_video_discriminator.backward()
        opt_video_discriminator.step()

        # reset gradients on generator
        opt_generator.zero_grad()

        # compute video generator losses
        generated_image_labels = image_discriminator(generated_image_batch)
        generated_video_labels = video_discriminator(generated_video_batch)

        # build generator ground truth 
        ones_image = self.ones_like(generated_image_labels)
        ones_video = self.ones_like(generated_video_labels)
        
        # compute generator loss
        # TODO: try reduction 'sum'
        l_generator = self.gan_criterion(generated_image_labels, ones_image) + \
                      self.gan_criterion(generated_video_labels, ones_video)
        
        l_reconstruction = 0.5 * F.l1_loss(generated_video_batch, real_video_batch[:,:,1:,:]) + \
                           0.5 * F.l1_loss(generated_image_batch, real_image_batch)
        l_generator += l_reconstruction
        
        # update generator
        l_generator.backward()
        opt_generator.step()

        return l_image_discriminator, l_video_discriminator, l_generator, l_reconstruction

    def train(self, generator, image_discriminator, video_discriminator, content_encoder, motion_encoder):
        if self.use_cuda:
            video_discriminator.cuda()

        logger = Logger(self.log_folder)

        # create optimizers
        opt_generator = optim.Adam(list(generator.parameters()) + list(content_encoder.parameters()) + list(motion_encoder.parameters()), 
                                   lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        opt_image_discriminator = optim.Adam(image_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)
        opt_video_discriminator = optim.Adam(video_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)

        # training loop

        def init_logs():
            return {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}

        batch_num = 0

        logs = init_logs()

        start_time = time.time()

        while True:
            generator.train()
            image_discriminator.train()
            video_discriminator.train()
            motion_encoder.train()
            content_encoder.train()

            opt_generator.zero_grad()

            opt_video_discriminator.zero_grad()

            (l_image, l_video, 
             l_gen, l_recon) = self.train_adversarial(generator, image_discriminator, video_discriminator, 
                                                      content_encoder, motion_encoder,
                                                      opt_image_discriminator, opt_video_discriminator, opt_generator)

            logs['l_gen'] += l_gen.item()
            logs['l_image_dis'] += l_image.item()
            logs['l_video_dis'] += l_video.item()
            logs['l_recon'] += l_recon.item()

            batch_num += 1

            if batch_num % self.log_interval == 0:

                log_string = "Batch %d" % batch_num
                for k, v in logs.iteritems():
                    log_string += " [%s] %5.3f" % (k, v / self.log_interval)

                log_string += ". Took %5.2f" % (time.time() - start_time)

                print log_string

                for tag, value in logs.items():
                    logger.scalar_summary(tag, value / self.log_interval, batch_num)

                logs = init_logs()
                start_time = time.time()

                generator.eval()
                motion_encoder.eval()
                content_encoder.eval()

                # images

                real_image_batch = self.sample_real_image_batch()
                latent_content = content_encoder(real_image_batch)
                latent_motion = motion_encoder(real_image_batch)
                images = generator.generate_images(latent_content, latent_motion)
                logger.image_summary("Images", images_to_numpy(images), batch_num)
                
                # video

                real_video_batch = self.sample_real_video_batch()
                real_video_batch_perm = real_video_batch.permute(0, 2, 1, 3, 4)
                latent_content = content_encoder(real_video_batch_perm[:,0,:])
                latent_motion = motion_encoder(real_video_batch_perm[:,0,:])
                videos = generator.generate_videos(latent_content, latent_motion, autoregressive=True)
                logger.video_summary("Videos", videos_to_numpy(videos.permute(0,2,1,3,4)), batch_num)
                torch.save(generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))

            if batch_num >= self.train_batches:
                torch.save(generator, os.path.join(self.log_folder, 'generator_%05d.pytorch' % batch_num))
                break
