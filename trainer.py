# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF


import json
import matplotlib.pyplot as plt

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import time
from PIL import Image


class Trainer:
    def __init__(self, options):
        self.opt = options
        
        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)
        self.log_path = os.path.join(self.opt.log_dir, current_time)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         'tartanair_depth': datasets.TartanAirDepthDataset,
                         'tartanair_odom': datasets.TartanAirOdomDataset,
                         'realsense_depth': datasets.RealSenseDepth}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            os.path.join(self.opt.data_path, 'train'), train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.calib, self.opt.min_depth, self.opt.max_depth, self.opt.data_aug, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            os.path.join(self.opt.data_path, 'val'), val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, self.opt.calib, self.opt.min_depth, self.opt.max_depth, False, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle = False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join('runs', current_time + '_' + mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()
        timings = {"backprop":0,"loader":0,"forward_time":0,"generate_time":0,"loss_time":0}
        before_data_time = time.time()
        for batch_idx, inputs in enumerate(self.train_loader):
            timings["loader"] += time.time() - before_data_time
            before_op_time = time.time()

            outputs, losses, timings = self.process_batch(inputs, timings)

            before_backprop_time = time.time()
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            timings['backprop'] += time.time() - before_backprop_time

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses, timings)
                self.val()

            self.step += 1
            before_data_time = time.time()

        self.model_lr_scheduler.step()

    def process_batch(self, inputs, timings, mode = 'train'):
        """Pass a minibatch through the network and generate images and losses
        """


        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.data_aug and mode == 'train':
            disp_aug = inputs["disp_aug", 0, 0]
            rgb_aug = inputs["rgb_aug", 0, 0]

            b,c,h,w = disp_aug.shape

            # rotate and flip
            hflip = torch.tensor(np.random.binomial(1, 0.5, b)).to(self.device)
            theta = torch.tensor(np.random.uniform(-5.0, 5.0, b) * np.pi / 180.0).to(self.device)

            # Flipping
            rgb_aug = hflip_img(rgb_aug, hflip, torch.float)
            disp_aug = hflip_img(disp_aug, hflip, torch.float)

            # Rotation
            rgb_aug = rot_img(rgb_aug, theta, torch.float)
            disp_aug = rot_img(disp_aug, theta, torch.float)
        else:
            disp_aug = inputs["disp", 0, 0]
            rgb_aug = inputs["rgb", 0, 0]

        before_forward_time = time.time()

        # Refine depth measurements
        features = self.models["encoder"](disp_aug, rgb_aug)
        outputs = self.models["depth"](features)

        if self.opt.data_aug and mode == 'train':
            
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]

                # Flipping
                disp = hflip_img(disp, hflip, torch.float)

                # rotation mask
                mask = torch.ones_like(disp)
                mask = rot_img(mask, -theta, torch.float)
                
                # Rotation
                disp = rot_img(disp, -theta, torch.float)

                outputs[("disp", scale)] = disp
                outputs[("mask", scale)] = mask

        timings["forward_time"] += time.time() - before_forward_time

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        before_generate_im = time.time()
        self.generate_images_pred(inputs, outputs)
        timings["generate_time"] += time.time() - before_generate_im

        before_compute_loss = time.time()
        losses = self.compute_losses(inputs, outputs)
        timings["loss_time"] += time.time() - before_compute_loss

        return outputs, losses, timings

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        timings = {"backprop":0,"loader":0,"forward_time":0,"generate_time":0,"loss_time":0}
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses, timings = self.process_batch(inputs, timings, mode='val')

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses, timings)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            if ('mask', scale) in outputs:
                mask = outputs[('mask', scale)]

            if self.opt.v1_multiscale: # use monodepthv1 multidepth
                source_scale = scale
            else: # use monodepthv2 multidepth
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            # backproject depth from depth (use rgb intrinsics as we have aligned depth with rgb)
            K_inv = inputs[("inv_K_rgb", source_scale)]
            cam_points = self.backproject_depth[source_scale](depth, K_inv)

            # project into ir0
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "ir0", t = 1)
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "ir0", t = 0)
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "ir0", t = -1)
             
            # project into ir1
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "ir1", t = 1)
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "ir1", t = 0)
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "ir1", t = -1)

            # project into rgb
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "rgb", t = 1)
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "rgb", t = 0)
            outputs = self.project_and_sample(inputs, cam_points, outputs, source_scale, scale, "rgb", t = -1)

    def project_and_sample(self, inputs, cam_points, outputs, source_scale, scale, cam, t):

        K = inputs[("K_{}".format(cam), source_scale)]
        T_cam = inputs[("T_{}".format(cam), t)]

        # T_RGB_C = inv(T_WB @ T_BS @ T_SRGB) @ (T_WB @ T_BS @ T_SC)
        # From RGB to cam
        T = torch.matmul(torch.inverse(T_cam), inputs[("T_rgb",0)])

        pix_coords = self.project_3d[source_scale](cam_points, K, T)

        # store intermediate value
        outputs[("sample_{}".format(cam), t, scale)] = pix_coords

        # get intensenties
        outputs[("{}".format(cam), t, scale)] = F.grid_sample(
            inputs[("{}".format(cam), t, source_scale)],
            outputs[("sample_{}".format(cam), t, scale)],
            padding_mode="border", align_corners=False)
        
        return outputs


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("rgb", 0, scale)]

            # t = 0: stereo
            target = outputs[("ir0", 0, source_scale)]
            pred = outputs[("ir1", 0, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_ir0_ir1",0,scale)] = reprojection_loss.clone()

            # t = -1: stereo
            target = outputs[("ir0", -1, source_scale)]
            pred = outputs[("ir1", -1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_ir0_ir1",-1,scale)] = reprojection_loss.clone()

            # t = 1: stereo
            target = outputs[("ir0", 1, source_scale)]
            pred = outputs[("ir1", 1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_ir0_ir1",1,scale)] = reprojection_loss.clone()

            # t = -1 --> t = 1 : stereo + temporal
            target = outputs[("ir0", -1, source_scale)]
            pred = outputs[("ir1", 1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_tmp_ir0_ir1",1,scale)] = reprojection_loss.clone()

            # t = -1 --> t = 1 : stereo + temporal
            target = outputs[("ir1", -1, source_scale)]
            pred = outputs[("ir0", 1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_tmp_ir1_ir0",1,scale)] = reprojection_loss.clone()

            # t = -1 --> t = 1 : temporal
            target = outputs[("ir0", -1, source_scale)]
            pred = outputs[("ir0", 1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_tmp_ir0_ir0",0,scale)] = reprojection_loss.clone()

            # t = -1 --> t = 1 : temporal
            target = outputs[("ir1", -1, source_scale)]
            pred = outputs[("ir1", 1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_tmp_ir1_ir1",0,scale)] = reprojection_loss.clone()

            # t = 0 --> t = -1 : temporal (rgb)
            target = outputs[("rgb", 0, source_scale)]
            pred = outputs[("rgb", -1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_tmp_rgb",1,scale)] = reprojection_loss.clone()

            # t = 0 --> t = 1 : temporal (rgb)
            target = outputs[("rgb", 0, source_scale)]
            pred = outputs[("rgb", 1, scale)]
            reprojection_loss = self.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reprojection_loss)
            outputs[("reproject_tmp_rgb",-1,scale)] = reprojection_loss.clone()

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                # they argue that relying on the min reduces artifacts at occlusions
                reprojection_loss, _ = torch.min(reprojection_loss, dim=1)

            reproject_loss = reprojection_loss.mean()

            # smoothness loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            outputs[('smooth_loss',0, scale)] = smooth_loss.clone()
            smooth_loss = self.opt.disparity_smoothness * smooth_loss.mean() / (2 ** scale)

            # input output loss
            inputoutput_loss = get_inputoutput_loss(disp, inputs[("disp", 0, scale)])
            outputs[('inputoutput_loss',0, scale)] = inputoutput_loss.clone()
            inputoutput_loss = self.opt.inputoutput_weight * inputoutput_loss.mean() / (2 ** scale)
            
            # accumulate losses
            loss = reproject_loss + smooth_loss + inputoutput_loss 
            total_loss += loss

            # only for logging purposes
            losses["loss/{}".format(scale)] = loss
            losses["reproject_loss/{}".format(scale)] = reproject_loss
            losses["smooth_loss/{}".format(scale)] = smooth_loss
            losses["inputoutput_loss/{}".format(scale)] = inputoutput_loss

            losses["reproject_prev_rgb/{}".format(scale)] = torch.mean(outputs[("reproject_tmp_rgb",-1,scale)])
            losses["reproject_next_rgb/{}".format(scale)] = torch.mean(outputs[("reproject_tmp_rgb",1,scale)])

            losses["reproject_tmp_ir0_ir0/{}".format(scale)] = torch.mean(outputs[("reproject_tmp_ir0_ir0",0,scale)])
            losses["reproject_tmp_ir1_ir1/{}".format(scale)] = torch.mean(outputs[("reproject_tmp_ir1_ir1",0,scale)])

            losses["reproject_tmp_ir1_ir0/{}".format(scale)] = torch.mean(outputs[("reproject_tmp_ir1_ir0",1,scale)])
            losses["reproject_tmp_ir0_ir1/{}".format(scale)] = torch.mean(outputs[("reproject_tmp_ir0_ir1",1,scale)])
            
            losses["reproject_next_ir0_ir1/{}".format(scale)] = torch.mean(outputs[("reproject_ir0_ir1",1,scale)])
            losses["reproject_curr_ir0_ir1/{}".format(scale)] = torch.mean(outputs[("reproject_ir0_ir1",0,scale)])
            losses["reproject_prev_ir0_ir1/{}".format(scale)] = torch.mean(outputs[("reproject_ir0_ir1",-1,scale)])

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [480, 640], mode="bilinear", align_corners=False), self.opt.min_depth, self.opt.max_depth)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        #crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {} | step: {} "
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left),self.step))



    def log(self, mode, inputs, outputs, losses, timings = None):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        if timings is not None:
            for l, v in timings.items():
                writer.add_scalar("time/{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            
            input_disp = inputs[("disp", 0, 0)][j]

            """
            import matplotlib.pyplot as plt
            plt.imshow(input_disp.detach().cpu().numpy()[0])
            plt.title("disp")
            plt.show()
            """
            
            _, input_depth = disp_to_depth(input_disp, self.opt.min_depth, self.opt.max_depth)

            """
            plt.imshow(input_depth.detach().cpu().numpy()[0])
            plt.title("input_depth_{}/{}".format(0, j))
            plt.show()
            """
            writer.add_image("input_depth_{}/{}".format(0, j), normalize_image(input_depth), self.step)

            #plt.figure(figsize=(15,5)); plt.subplot(1,3,1)
            #plt.imshow(input_disp[0].cpu().numpy()); plt.title("disp"); plt.subplot(1,3,2)
            #plt.imshow(input_depth[0].cpu().numpy(), vmin=self.opt.min_depth,vmax=self.opt.max_depth);  plt.title("depth"); plt.subplot(1,3,3)
            #plt.imshow(normalize_image(input_depth, self.opt.max_depth, self.opt.min_depth)[0].cpu().numpy()); plt.title("norm depth"); plt.show()

            for s in [0]:#self.opt.scales:

                ###
                # Inputs and predictions
                ###
                
                # add ir images
                writer.add_image(
                    "input_ir0_curr_{}_{}/{}".format(0, s, j),
                    inputs[("ir0", 0, s)][j].data, self.step)
                writer.add_image(
                    "input_ir1_curr_{}_{}/{}".format(0, s, j),
                    inputs[("ir1", 0, s)][j].data, self.step)
                writer.add_image(
                    "input_ir0_prev_{}_{}/{}".format(0, s, j),
                    inputs[("ir0", -1, s)][j].data, self.step)
                writer.add_image(
                    "input_ir1_prev_{}_{}/{}".format(0, s, j),
                    inputs[("ir1", -1, s)][j].data, self.step)
                writer.add_image(
                    "input_ir0_next_{}_{}/{}".format(0, s, j),
                    inputs[("ir0", 1, s)][j].data, self.step)
                writer.add_image(
                    "input_ir1_next_{}_{}/{}".format(0, s, j),
                    inputs[("ir1", 1, s)][j].data, self.step)

                # add color image
                writer.add_image(
                    "input_color_curr_{}_{}/{}".format(0, s, j),
                    inputs[("rgb", 0, s)][j].data, self.step)
                writer.add_image(
                    "input_color_prev_{}_{}/{}".format(0, s, j),
                    inputs[("rgb", -1, s)][j].data, self.step)
                writer.add_image(
                    "input_color_next_{}_{}/{}".format(0, s, j),
                    inputs[("rgb", 1, s)][j].data, self.step)

                # add prediction
                writer.add_image(
                    "pred_depth_{}/{}".format(s, j),
                    normalize_image(outputs[("depth", 0, s)][j]), self.step)

                ###
                # Warpings
                ###

                # add ir reprojections images
                writer.add_image(
                    "output_ir0_curr_{}_{}/{}".format(0, s, j),
                    outputs[("ir0", 0, s)][j].data, self.step)
                writer.add_image(
                    "output_ir1_curr_{}_{}/{}".format(0, s, j),
                    outputs[("ir1", 0, s)][j].data, self.step)

                writer.add_image(
                    "output_ir0_prev_{}_{}/{}".format(0, s, j),
                    outputs[("ir0", -1, s)][j].data, self.step)
                writer.add_image(
                    "output_ir1_prev_{}_{}/{}".format(0, s, j),
                    outputs[("ir1", -1, s)][j].data, self.step)

                writer.add_image(
                    "output_ir0_next_{}_{}/{}".format(0, s, j),
                    outputs[("ir0", 1, s)][j].data, self.step)
                writer.add_image(
                    "output_ir1_next_{}_{}/{}".format(0, s, j),
                    outputs[("ir1", 1, s)][j].data, self.step)

                # add rgb reprojection images
                writer.add_image(
                    "output_rgb_next_{}_{}/{}".format(0, s, j),
                    outputs[("rgb", 1, s)][j].data, self.step)
                writer.add_image(
                    "output_rgb_prev_{}_{}/{}".format(0, s, j),
                    outputs[("rgb", -1, s)][j].data, self.step)


                ###
                # Losses
                ###

                # add photometric loss ir0 => ir1
                writer.add_image(
                    "reproject_stereo_curr_{}/{}".format(s, j),
                    normalize_image(outputs[("reproject_ir0_ir1", 0, s)][j]), self.step)

                # add photometric loss ir0 => ir1
                writer.add_image(
                    "reproject_stereo_next_{}/{}".format(s, j),
                    normalize_image(outputs[("reproject_ir0_ir1", 1, s)][j]), self.step)

                # add photometric loss ir0 => ir1
                writer.add_image(
                    "reproject_stereo_prev_{}/{}".format(s, j),
                    normalize_image(outputs[("reproject_ir0_ir1", -1, s)][j]), self.step)

                # add photometric loss rgb => rgb
                writer.add_image(
                    "reproject_rgb_next_{}/{}".format(s, j),
                    normalize_image(outputs[("reproject_tmp_rgb", 1, s)][j]), self.step)

                # add photometric loss rgb => rgb
                writer.add_image(
                    "reproject_rgb_prev_{}/{}".format(s, j),
                    normalize_image(outputs[("reproject_tmp_rgb", -1, s)][j]), self.step)

                # add smoothness loss
                writer.add_image(
                    "smooth_loss_{}/{}".format(s, j),
                    normalize_image(outputs[("smooth_loss", 0, s)][j]), self.step)
                
                # add input/output loss
                writer.add_image(
                    "inputoutput_loss_{}/{}".format(s, j),
                    normalize_image(outputs[("inputoutput_loss", 0, s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                #elif not self.opt.disable_automasking:
                #    writer.add_image(
                #        "automask_{}/{}".format(s, j),
                #        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
