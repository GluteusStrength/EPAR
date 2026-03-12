'''
Following M3DM(CVPR 2023), we composed dataset for MVTec3D and Eyecandies.
'''

import os
import random
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import resize_organized_pc, read_tiff_organized_pc, organized_pc_to_depth_map # resized_organized_pc: resize point clouds
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils import point_transforms as point_transforms
import torchvision.transforms as T
from perlin_noise import PerlinNoise

# Eyecandies classes
def eyecandies_classes():
    return [
        'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy',   
    ]

# MVTec3D classes
def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]

RGB_SIZE = 224
DARK_THRESHOLD = 0.2

class AnomalyDetectionDataset(Dataset):
    def __init__(self, split, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406] # ImageNet mean
        self.IMAGENET_STD = [0.229, 0.224, 0.225] # ImageNet std
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(dataset_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
        self.grayscale_transform = T.Grayscale(num_output_channels=1)

class TrainDataset(AnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path, dtd_dataset_path="/home/ijaehojo/AD-JEPA/datasets/dtd/", patch_size=8, k_shot=-1): 
        super().__init__(split="train", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.k_shot = k_shot
        self.img_paths, self.labels = self.load_dataset(self.k_shot)
        self.patch_size = patch_size
        self.img_size = img_size
        # --- DTD dataset for RGB Anomaly Synthesis ---
        if dtd_dataset_path:
            self.dtd_paths = glob.glob(os.path.join(dtd_dataset_path, 'images', '**', '*.jpg'), recursive=True)
            self.dtd_paths.sort() # Fix the order
            print(f"Loaded {len(self.dtd_paths)} DTD images.")
        else:
            self.dtd_paths = []
            print("Warning: DTD dataset path not provided. Anomaly generation will not use DTD.")
        self.dtd_gray_transform = T.Grayscale(num_output_channels=3)
       
        # DTD Augmentation
        transform_list = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=(-45, 45)),
            # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ]
        self.source_transform = T.Compose(transform_list)

    def load_dataset(self, k_shot):
        img_tot_paths, tot_labels = [], []
        glob_pattern = os.path.join(self.img_path, 'good', 'rgb') + "/*.png"
        rgb_paths = glob.glob(glob_pattern)
        if k_shot != -1:
            rgb_paths = rgb_paths[:k_shot]
            
        tiff_pattern = os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff"
        tiff_paths = glob.glob(tiff_pattern)
        if k_shot != -1:
            tiff_paths = tiff_paths[:k_shot]
            
        rgb_paths.sort(); tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels
    
    # Like several prior works, apply perlin mask.
    def generate_perlin_mask(self, size=(224, 224), threshold=0.5, scale=25, octaves=4):
        noise_gen = PerlinNoise(octaves=octaves, seed=np.random.randint(10000))
        mask = np.zeros(size)

        for i in range(size[0]):
            for j in range(size[1]):
                mask[i, j] = noise_gen([i / scale, j / scale])

        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask = (mask > threshold).astype(np.float32)
        
        return torch.tensor(mask).unsqueeze(0)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, tiff_path = self.img_paths[idx]
        label = self.labels[idx] # 0: Good
        
        raw_img = Image.open(rgb_path).convert('RGB')
        img_tensor = self.rgb_transform(raw_img) # [3, H, W] (원본 RGB)
        
        organized_pc = read_tiff_organized_pc(tiff_path)   # (H, W, 3)
        # Original Depth
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        depth_tensor = resized_depth_map_3channel.clone().detach().float() 
        # Original 3D Point Cloud
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.img_size, target_width=self.img_size).clone().detach().float()
            
        return {
            'data': (img_tensor, resized_organized_pc, depth_tensor, label),
            'label': label 
        }

class ValidationDataset(AnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path, dtd_dataset_path="/home/ijaehojo/AD-JEPA/datasets/dtd/", patch_size=8, k_shot=-1): 
        super().__init__(split="validation", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.k_shot = k_shot
        self.img_paths, self.labels = self.load_dataset(self.k_shot)
        self.patch_size = patch_size
        self.img_size = img_size
        # --- DTD dataset for RGB Anomaly Synthesis ---
        if dtd_dataset_path:
            self.dtd_paths = glob.glob(os.path.join(dtd_dataset_path, 'images', '**', '*.jpg'), recursive=True)
            self.dtd_paths.sort() # Fix the order
            print(f"Loaded {len(self.dtd_paths)} DTD images.")
        else:
            self.dtd_paths = []
            print("Warning: DTD dataset path not provided. Anomaly generation will not use DTD.")
        self.dtd_gray_transform = T.Grayscale(num_output_channels=3)
       
        # DTD Augmentation
        transform_list = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=(-45, 45)),
            # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ]
        self.source_transform = T.Compose(transform_list)

    def load_dataset(self, k_shot):
        img_tot_paths, tot_labels = [], []
        glob_pattern = os.path.join(self.img_path, 'good', 'rgb') + "/*.png"
        rgb_paths = glob.glob(glob_pattern)
        if k_shot != -1:
            rgb_paths = rgb_paths[:k_shot]
            
        tiff_pattern = os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff"
        tiff_paths = glob.glob(tiff_pattern)
        if k_shot != -1:
            tiff_paths = tiff_paths[:k_shot]
            
        rgb_paths.sort(); tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels
    
    # Like several prior works, apply perlin mask.
    def generate_perlin_mask(self, size=(224, 224), threshold=0.5, scale=25, octaves=4):
        noise_gen = PerlinNoise(octaves=octaves, seed=np.random.randint(10000))
        mask = np.zeros(size)

        for i in range(size[0]):
            for j in range(size[1]):
                mask[i, j] = noise_gen([i / scale, j / scale])

        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask = (mask > threshold).astype(np.float32)
        
        return torch.tensor(mask).unsqueeze(0)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, tiff_path = self.img_paths[idx]
        label = self.labels[idx] # 0: Good
        
        raw_img = Image.open(rgb_path).convert('RGB')
        img_tensor = self.rgb_transform(raw_img) # [3, H, W] (원본 RGB)
        
        organized_pc = read_tiff_organized_pc(tiff_path)   # (H, W, 3)
        # Original Depth
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        depth_tensor = resized_depth_map_3channel.clone().detach().float() 
        # Original 3D Point Cloud
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.img_size, target_width=self.img_size).clone().detach().float()
            
        return {
            'data': (img_tensor, resized_organized_pc, depth_tensor, label),
            'label': label 
        }

    
class TestDataset(AnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(split="test", class_name=class_name, img_size=img_size, dataset_path=dataset_path)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = [] 
        defect_types = os.listdir(self.img_path) # load defect types

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)
        # Grayscale tensor
        gray_tensor = self.grayscale_transform(img)
        object_mask = (gray_tensor > DARK_THRESHOLD).float() # [1, H, W]
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.size, target_width=self.size)
        resized_organized_pc = resized_organized_pc.clone().detach().float()
        
        if gt == 0: # normal
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else: # abnormal 
            gt = Image.open(gt).convert('L') # gray scale images
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return {'data': (img, resized_organized_pc, resized_depth_map_3channel, object_mask), 
                'anomaly_map': gt[:1],
                'label': label,
                'rgb_img_pth': rgb_path}
