# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import pandas as pd
import cv2
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
#from kitti_utils import generate_depth_map
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import yaml

from layers import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_intrinsic_extrinsic(data):

    T = np.asarray(data['T_SC']).reshape(4,4).astype(dtype=np.float32)
    
    dis_coeff = data['distortion_coefficients']
    F = data['focal_length']
    pp = data['principal_point']
    dim = data['image_dimension']

    K_normalized = np.asarray([[F[0]/dim[0], 0, pp[0]/dim[0], 0],
                    [0, F[1]/dim[1], pp[1]/dim[1], 0],
                    [0,0,1,0],
                    [0,0,0,1]]).astype(dtype=np.float32)


    K = np.asarray([[F[0], 0, pp[0], 0],
                    [0, F[1], pp[1], 0],
                    [0,0,1,0],
                    [0,0,0,1]]).astype(dtype=np.float32)

    dis_coeff = np.asarray(dis_coeff).reshape(4,1)

    return K, K_normalized, T, dis_coeff

class RealSenseDepth(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 calibration_file,
                 min_depth, 
                 max_depth,
                 data_aug,
                 is_train=False,
                 img_ext='.jpg'):
        super(RealSenseDepth, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        with open(calibration_file) as file:
            calib = yaml.load(file, Loader=yaml.FullLoader)

        self.offset = calib["camera_params"]["timeshift_cam_imu"]
        
        # Inferred0
        self.K_ir0, self.K_norm_ir0, self.T_ir0, _ = get_intrinsic_extrinsic(calib['cameras'][0])

        # Inferred1
        self.K_ir1, self.K_norm_ir1, self.T_ir1, _ = get_intrinsic_extrinsic(calib['cameras'][1])

        # Depth
        self.K_dep, self.K_norm_dep, self.T_dep, _ = get_intrinsic_extrinsic(calib['cameras'][2])

        # Visible
        self.K_rgb, self.K_norm_rgb, self.T_rgb, self.dis_coeff_rgb = get_intrinsic_extrinsic(calib['cameras'][3])

        # load data
        self.data, self.index = self.load_data(data_path)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.disp_resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.disp_resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=Image.NEAREST)

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

        self.random_erase = transforms.RandomErasing()
        self.data_aug = data_aug

        self.device = torch.device("cpu" if False else "cuda")
        self.backproject_depth = BackprojectDepth(1, height, width)
       # self.backproject_depth.to(self.device)

        self.project_3d = Project3D(1, height, width)
        #self.project_3d.to(self.device)

    def preprocess(self, inputs):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars

        for k in list(inputs):
            frame = inputs[k]
            if "K_" not in k or 'T_' not in k:
                n, im, i = k
                for i in range(self.num_scales):

                    if n == 'disp':
                        inputs[(n, im, i)] = self.disp_resize[i](inputs[(n, im, i - 1)])
                    else:
                        inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "K_" not in k or 'T_' not in k:
                n, im, i = k
                                
                inputs[(n, im, i)] = self.to_tensor(np.asarray(f).copy())
                
                if self.data_aug:
                    if n == 'rgb' and im == 0: # we only want to augment this center image
                        inputs[(n + "_aug", im, i)] = self.to_tensor(self.color_aug(f).copy())

                    if n == 'disp' and im == 0: # we only want to augment this center image
                        inputs[(n + "_aug", im, i)] = self.random_erase(self.to_tensor(np.asarray(f).copy()))

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            
            # RGB
            K_rgb = self.K_norm_rgb.copy()

            K_rgb[0, :] *= self.width // (2 ** scale)
            K_rgb[1, :] *= self.height // (2 ** scale)

            inv_K_rgb = np.linalg.pinv(K_rgb)

            inputs[("K_rgb", scale)] = torch.from_numpy(K_rgb)
            inputs[("inv_K_rgb", scale)] = torch.from_numpy(inv_K_rgb)

            # Depth
            K_dep = self.K_norm_dep.copy()

            K_dep[0, :] *= self.width // (2 ** scale)
            K_dep[1, :] *= self.height // (2 ** scale)

            inv_K_dep = np.linalg.pinv(K_dep)

            inputs[("K_dep", scale)] = torch.from_numpy(K_dep)
            inputs[("inv_K_dep", scale)] = torch.from_numpy(inv_K_dep)

            # IR0
            K_ir0 = self.K_norm_ir0.copy()

            K_ir0[0, :] *= self.width // (2 ** scale)
            K_ir0[1, :] *= self.height // (2 ** scale)

            inv_K_ir0 = np.linalg.pinv(K_ir0)

            inputs[("K_ir0", scale)] = torch.from_numpy(K_ir0)
            inputs[("inv_K_ir0", scale)] = torch.from_numpy(inv_K_ir0)

            # IR1
            K_ir1 = self.K_norm_ir1.copy()

            K_ir1[0, :] *= self.width // (2 ** scale)
            K_ir1[1, :] *= self.height // (2 ** scale)

            inv_K_ir1 = np.linalg.pinv(K_ir1)

            inputs[("K_ir1", scale)] = torch.from_numpy(K_ir1)
            inputs[("inv_K_ir1", scale)] = torch.from_numpy(inv_K_ir1)

        return inputs

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("ir_r", <frame_id>, <scale>)          for right infered image,
            ("ir_l", <frame_id>, <scale>)          for left infered image,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("depth_aug", <frame_id>, <scale>)      for augmented depth images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """

        frame_idx = self.index[index]

        inputs = {}

        # make sure that projector is only on in the current frame
        proj_intensity_t0 = self.data[frame_idx, -17]
        proj_intensity_tm1 = self.data[frame_idx - 1, -17]
        proj_intensity_tp1 = self.data[frame_idx + 1, -17]

        if proj_intensity_t0 != 150 or proj_intensity_tm1 != 0 or proj_intensity_tp1 != 0:
            print("error!")
            print("fame idx", frame_idx)
            print()
            print(self.data[frame_idx, :])
            print(proj_intensity_t0, proj_intensity_tm1, proj_intensity_tp1)
        
        inputs[("disp", 0, -1)] = self.load_disp(self.data[frame_idx, 3])

        inputs[("rgb", 0, -1)] = self.undistort_rgb(self.load_im(self.data[frame_idx, 1]))
        inputs[("rgb", -1, -1)] = self.undistort_rgb(self.load_im(self.data[frame_idx - 1, 1]))
        inputs[("rgb", 1, -1)] = self.undistort_rgb(self.load_im(self.data[frame_idx + 1, 1]))

        inputs[("ir0", 0, -1)] = self.load_im(self.data[frame_idx, 4])
        inputs[("ir0", -1, -1)] = self.load_im(self.data[frame_idx - 1, 4])
        inputs[("ir0", 1, -1)] = self.load_im(self.data[frame_idx + 1, 4])

        inputs[("ir1", 0, -1)] = self.load_im(self.data[frame_idx, 5])
        inputs[("ir1", -1, -1)] = self.load_im(self.data[frame_idx - 1, 5])
        inputs[("ir1", 1, -1)] = self.load_im(self.data[frame_idx + 1, 5])
        
        T_WB_prev = self.data[frame_idx - 1, -16:].reshape(4,4)
        T_WB_curr = self.data[frame_idx, -16:].reshape(4,4)
        T_WB_next = self.data[frame_idx + 1, -16:].reshape(4,4)
        T_BS = np.diag(np.ones(4))

        inputs= self.preprocess(inputs)

        # camera to world (T_WC = T_WB @ T_BS @ T_SC)
        inputs[('T_ir0', 0)] = np.matmul(np.matmul(T_WB_curr, T_BS), self.T_ir0).astype(dtype=np.float32)
        inputs[('T_ir0', -1)] = np.matmul(np.matmul(T_WB_prev, T_BS), self.T_ir0).astype(dtype=np.float32)
        inputs[('T_ir0', 1)] = np.matmul(np.matmul(T_WB_next, T_BS), self.T_ir0).astype(dtype=np.float32)
        
        inputs[('T_ir1',0)] = np.matmul(np.matmul(T_WB_curr, T_BS), self.T_ir1).astype(dtype=np.float32)
        inputs[('T_ir1',-1)] = np.matmul(np.matmul(T_WB_prev, T_BS), self.T_ir1).astype(dtype=np.float32)
        inputs[('T_ir1',1)] = np.matmul(np.matmul(T_WB_next, T_BS), self.T_ir1).astype(dtype=np.float32)

        inputs['T_dep'] = np.matmul(np.matmul(T_WB_curr, T_BS), self.T_dep).astype(dtype=np.float32)

        inputs[('T_rgb', 0)] = np.matmul(np.matmul(T_WB_curr, T_BS), self.T_rgb).astype(dtype=np.float32)
        inputs[('T_rgb', -1)] = np.matmul(np.matmul(T_WB_prev, T_BS), self.T_rgb).astype(dtype=np.float32)
        inputs[('T_rgb', 1)] = np.matmul(np.matmul(T_WB_next, T_BS), self.T_rgb).astype(dtype=np.float32)

        return inputs

    def calc_diff(self, frame1, frame0):
        # calulate transformation from frame0 to frame1
        #        
        t0 = frame0[1:4]
        q0 = R.from_quat(frame0[4:])

        t1 = frame1[1:4]
        q1 = R.from_quat(frame1[4:])

        # relative translation
        t = t1 - t0

        # relative rotation
        q1_inv = q1.inv()
        q = q0 * q1_inv

        # form quaternioms to matrix
        R = q.as_matrix()

        T = np.diag(4)
        T[:3, :3] = R
        T[3:, :3] = t

        return T

    def load_seq(self, datapath):

        data = {}
        _lengths = []
        for sensor in ['cam0', 'depth0', 'ir0', 'ir1']:
            data[sensor] = pd.read_csv(os.path.join(datapath,sensor,"data.csv")).values
            _lengths.append(len(data[sensor]))

        # make sure they have equal length
        for sensor in ['cam0', 'depth0', 'ir0', 'ir1']:
            data[sensor] = data[sensor][:min(_lengths)]
            data[sensor][:,1] = [os.path.join(datapath, sensor, 'data', fn) for fn in data[sensor][:,1]]


        # put everything into the same dataframe for ease
        # rgb time, rgb path, ir time, depth path, ir0 path, ir1 path, projector on
        dataframe = np.concatenate([data['cam0'][:, :2], data['depth0'][:, :2], data['ir0'][:, 1:2], data['ir1'][:, 1:2], data['ir1'][:, -1:]], axis=1)

        # load postion information
        T_WB = pd.read_csv(os.path.join(datapath, 'position', 'optimised_trajectory_0.csv')).values

        T_WB_interpolated, valid_idx = self.interpolate(T_WB, dataframe[:,2], self.offset)

        dataframe = np.concatenate([dataframe[valid_idx, :], T_WB_interpolated], axis=1) 

        # only with laser images
        idx = np.where(dataframe[:,-17] == 150)[0]

        # Find indices that have both image in front and after
        idx = idx[(idx > 0) * (idx < len(dataframe))]

        return dataframe, idx

    def load_data(self, datapath):
        
        index = []
        data = []
        for seq in os.listdir(datapath):

            seq_data, idx = self.load_seq(os.path.join(datapath, seq))
                        
            index.extend(idx + len(data))
            data = seq_data if len(data) == 0 else np.concatenate([data, seq_data], axis = 0) 
                                    
        return data, index

    def interpolate(self, T_WB, t_ir0, offset):

        t = T_WB[:,0].astype(float) - offset
        t_ir0 = t_ir0.astype(float)

        # times (find idx where we have between slam postion estimates)
        idx = np.where((t_ir0 > min(t)) * (t_ir0 < max(t)))[0]
        t_ir0_with_pos = t_ir0[idx]

        # interpolate  translation
        x = T_WB[:,1]
        y = T_WB[:,2]
        z = T_WB[:,3]

        f = interp1d(t, x)
        new_x = f(t_ir0_with_pos)

        f = interp1d(t, y)
        new_y = f(t_ir0_with_pos)

        f = interp1d(t, z)
        new_z = f(t_ir0_with_pos)

        # interpolate rotations
        q = T_WB[:,4:]
        q = R.from_quat(q)

        f = Slerp(t, q)
        q_new = f(t_ir0_with_pos)

        # initialize T
        T = np.diag(np.ones(4))
        T = np.repeat(T[None,:,:],len(t_ir0_with_pos), axis=0)

        # insert into T (here we have some padding to get the same length of the images)
        # This makes indexing in getitem significantly easier
        T[:,:3,:3] = q_new.as_matrix()
        T[:,0,3] = new_x
        T[:,1,3] = new_y
        T[:,2,3] = new_z

        # reshape T to fit into dataframe
        return T.reshape(-1, 16), idx

    def undistort_rgb(self, im):
        
        im = np.asarray(im)
        im = cv2.undistort(im, self.K_rgb[:3,:3], self.dis_coeff_rgb, None, self.K_rgb[:3,:3])
        im = Image.fromarray(im)

        return im

    def load_im(self, path):

        im = Image.open(path)
        #im = self.to_tensor(im)

        im = transforms.functional.crop(im, 0, 0, self.height, self.width)

        return im

    def load_depth(self, path):

        dep = Image.open(path)
        dep = transforms.functional.crop(dep, 0, 0, self.height, self.width)
        dep = np.array(dep)
        dep = dep.astype(np.float32) / (1000.0) 
        # dep stored in milimeters so dividing with 1000 gives meters

        #import matplotlib.pyplot as plt
        #plt.imshow(dep)
        #plt.show()


        return dep

    def backproject_depth_numpy(self, depth):
        
        depth = depth.flatten()

        fx = self.K_dep[0,0]
        fy = self.K_dep[1,1]
        cx = self.K_dep[0,2]
        cy = self.K_dep[1,2]

        # create x,y
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        pix_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        pix_coords = np.stack([pix_coords[0].flatten(), 
                                pix_coords[1].flatten(),
                                np.ones(self.height*self.width)]).T

        x = depth * (pix_coords[:, 0] - cx) / fx
        y = depth * (pix_coords[:, 1] - cy) / fy

        cam_points = np.stack([x, y, depth, 
                        np.ones((self.height*self.width))], axis=1)
        
        return cam_points

    def project_3d_numpy(self, Q):

        K_rgb = self.K_rgb[:3,:3]
        q2, J = cv2.projectPoints(Q[0:3,:].T,
                                np.zeros([1, 3]), 
                                np.zeros([1, 3]),
                                K_rgb,
                                self.dis_coeff_rgb) 

        # insert in image
        x_ = np.round(q2[:,0,0]).astype(int)
        y_ = np.round(q2[:,0,1]).astype(int)
        z = Q[2]
        
        # mask pixels outside camera plance
        mask = (x_ >= 0) * (x_ < self.width)
        mask *= (y_ >= 0) * (y_ < self.height) 
        mask *= z > 0
        mask = np.where(mask == 1)[0]

        aligned_depth = np.zeros((self.height, self.width))
        aligned_depth[y_[mask], x_[mask]] = z[mask]

        #aligned_depth = scipy.signal.medfilt2d(aligned_depth, (3,3)) 
        #aligned_depth = cv2.dilate(aligned_depth,(5,5),iterations = 1)

        return aligned_depth

    def align_depth_to_rgb(self, dep):

        """
        import matplotlib.pyplot as plt
        plt.subplot(1,3,1)
        plt.imshow(dep)
        plt.title("depth")
        plt.subplot(1,3,2)
        """

        # backproject depth from depth
        cam_points = self.backproject_depth_numpy(dep)

        # depth -> sensor -> rgb
        T = np.matmul(np.linalg.inv(self.T_rgb), self.T_dep)
        Q = np.matmul(T, cam_points.T)

        # project into rgb frame
        aligned_depth = self.project_3d_numpy(Q)

        """
        plt.imshow(aligned_depth)
        plt.title("aligned_depth")
        plt.subplot(1,3,3)
        """

        kernel = np.ones((3,3),np.uint8)
        aligned_depth = cv2.dilate(aligned_depth,kernel,iterations = 1)

        """
        plt.imshow(aligned_depth)
        plt.title("dilated")
        plt.show()
        """
        
        return aligned_depth

    def dep_to_disp(self, dep):

        mask = (dep > self.min_depth) * (dep < self.max_depth)

        disp = np.zeros_like(dep)
        disp[mask] = 1.0 / dep[mask]

        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth

        scaled_disp = (disp - min_disp) / (max_disp - min_disp)
        scaled_disp[mask==0] = 0

        scaled_disp_im = Image.fromarray(scaled_disp)

        return scaled_disp_im

    def load_disp(self, path):
        
        dep = self.load_depth(path)

        """
        import matplotlib.pyplot as plt
        plt.imshow(dep)
        plt.title("depth")
        plt.show()
        """
        
        dep = self.align_depth_to_rgb(dep)

        disp = self.dep_to_disp(dep)
        
        """
        import matplotlib.pyplot as plt
        plt.imshow(np.asarray(disp))
        plt.title("aligned_disp")
        plt.show()
        """

        return disp
