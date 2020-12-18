# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

from torch.utils.data import DataLoader
import datasets

import torch
from torchvision import transforms#, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--data_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--output_directory', type=str,
                        help='path to a output directory', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--calib', type=str,
                        help='path to calibration file')
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=3)
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')



    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.isdir(args.output_directory):
        for f in ['pred', 'input']:
            os.makedirs(os.path.join(args.output_directory, f))
    else:
        print("Warning! Overwriting output files")


    #download_model_if_doesnt_exist(args.model_name)
    model_path = args.model_name
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # DEFINE DATALOADER
    dataset = datasets.RealSenseDepth(args.data_path, None, feed_height, feed_width,
        None, 4, args.calib, args.min_depth, args.max_depth, is_train=False)

    dataloader = DataLoader(dataset, 1, shuffle = False, num_workers=4, pin_memory=True)

    print("-> Predicting on {:d} test images".format(len(dataset)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):

            input_disp = inputs["disp", 0, 0].to(device)
            rgb = inputs["rgb", 0, 0].to(device)

            # PREDICTION
            features = encoder(input_disp, rgb)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]

            # Saving numpy file
            output_name = str(idx).zfill(6)
            name_dest_npy = os.path.join(args.output_directory, 'pred', "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, args.min_depth, args.max_depth)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(args.output_directory, 'pred', "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            # Saving colormapped depth image
            disp_resized_np = input_disp.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(args.output_directory, 'input', "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(dataset), name_dest_im))

    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)


    """
    python test_simple.py --data_path '../../data/d455/val'  --output_directory '~/Desktop/results_16_12' --model_name 'pretrained/16_21_54/models/weights_19' --calib calibration/032522250355.yaml
    """