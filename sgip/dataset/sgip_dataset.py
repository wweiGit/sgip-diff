from typing import Sequence, Dict, Union, List, Mapping, Any, Optional, Callable
import math
import time
import io
import random
import os
import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
import json 
from .degradation import (
    random_mixed_kernels,
    random_add_gaussian_noise,
    random_add_jpg_compression,
)
from .utils import load_file_list, center_crop_arr, random_crop_arr
from ..utils.common import instantiate_from_config
import torch
from albumentations import Compose,RandomRotate90,HorizontalFlip,VerticalFlip
import albumentations as A

DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

class SgipDataset(data.Dataset):

    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
    ) -> "SgipDataset":
        super(SgipDataset, self).__init__()
        self.data = json.load(open(file_list))
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations

    def load_image(
        self, image_path: str, max_retry: int = 5, resize = False
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = center_crop_arr(image, self.out_size)
                elif self.crop_type == "random":
                    image = random_crop_arr(image, self.out_size, min_crop_frac=0.7)
        else:
            if resize:
                image = image.resize((self.out_size, self.out_size), Image.LANCZOS)
            # assert image.height == self.out_size and image.width == self.out_size
            image = np.array(image)
        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        item = self.data[index] 
        # read image
        prompt = "photo of a id person, ultra HD, highly detailed, skin pore detailing, perfect lighting, perfect front view."
        
        # prompt = "photo of a id person"
        
        img_lq = self.load_image(item["img_path"], resize = True)
        img_gt = self.load_image(item["gt_path"], resize = False)
        img_3dmm = np.array(Image.open(item["emoca_path"])).astype(np.float32)
        
        img_gt = (img_gt / 255.0).astype(np.float32)
        img_lq = (img_lq / 255.0).astype(np.float32)
        
        face_id_embed = torch.load(item["id_path"], map_location="cpu").detach()
        #print("face_id_embed requires_grad:", face_id_embed.requires_grad)
        face_id_embed = face_id_embed / torch.norm(face_id_embed)   # normalize embedding
        
        if item["clean_path"]:
            img_clean = self.load_image(item["clean_path"])
            img_clean = (img_clean / 255.0).astype(np.float32)
        else:
            img_clean = np.array([])
        
        img_gt = (img_gt * 2 - 1).astype(np.float32)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "clean": img_clean,
            "emoca": img_3dmm,
            "face_id_embed": face_id_embed,
            "text": prompt,
            "img_name": item["img_name"]
        }
        
    def __len__(self) -> int:
        return len(self.data)

aug_pipeline1 = Compose([
    HorizontalFlip(p=0.5)]) 

aug_pipeline2 = Compose([
        # 阶段1：几何形变 (模拟非正面)
        A.OneOf([
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.1, 0.1),
                shear=(-8, 8),
                p=0.5
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=10,
                p=0.3
            )
        ], p=0.5),
        # 阶段2：光照扰动
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.3),
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=30,  # 色相变化
                sat_shift_limit=30,  # 饱和度变化
                val_shift_limit=30,  # 明度变化
                p=0.5
            )
        ], p=0.5)])
        
class DatasetPair(data.Dataset):
    
    def __init__(
        self,
        file_list: str,
        file_backend_cfg: Mapping[str, Any],
        out_size: int,
        crop_type: str,
        gt_path: str,
        aug: str
    ) -> "DatasetPair":
        super(DatasetPair, self).__init__()
        self.data = json.load(open(file_list))
        self.file_backend = instantiate_from_config(file_backend_cfg)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.gt_path = gt_path
        if aug == "type1":
            self.augmentations = aug_pipeline1
        else:
            self.augmentations = None
        
    def load_image(
        self, image_path: str, resize: str, max_retry: int = 5 
    ) -> Optional[np.ndarray]:
        image_bytes = None
        while image_bytes is None:
            if max_retry == 0:
                return None
            image_bytes = self.file_backend.get(image_path)
            max_retry -= 1
            if image_bytes is None:
                time.sleep(0.5)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        if resize and (image.height != self.out_size or image.width != self.out_size):
            image = image.resize((self.out_size, self.out_size), Image.LANCZOS)
            
        image = np.array(image)
            

        # hwc, rgb, 0,255, uint8
        return image

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, str]]:
        # load gt image
        item = self.data[index] 
        img_lq = self.load_image(item["img_path"], resize = True)
        img_name = os.path.basename(item["img_path"])
        gt_path = os.path.join(self.gt_path, img_name)
        img_gt = self.load_image(gt_path, resize = False)
        
        img_3dmm = Image.open(item["emoca_path"])
        img_3dmm = img_3dmm.resize((self.out_size, self.out_size), Image.LANCZOS)
        img_3dmm = np.array(img_3dmm).astype(np.float32)
        
        #cv2.imwrite(os.path.join("/root/autodl-tmp/project/SGIP-dev-v2/work_dirs/tmp", f"img_lqo_{index}.png"), img_lq)
        
        # 应用数据增强  确保同时处理LQ和GT图像
        if self.augmentations is not None:
            augmented = self.augmentations(image=img_lq)
            img_lq = augmented["image"]

        #cv2.imwrite(os.path.join("/root/autodl-tmp/project/SGIP-dev-v2/work_dirs/tmp", f"img_lq_{index}.png"), img_lq)
        #cv2.imwrite(os.path.join("/root/autodl-tmp/project/SGIP-dev-v2/work_dirs/tmp", f"img_gt_{index}.png"), img_gt)
 
        img_gt = (img_gt / 255.0).astype(np.float32)
        img_lq = (img_lq / 255.0).astype(np.float32)
        img_gt = (img_gt * 2 - 1).astype(np.float32)

        return {
            "lq": img_lq,
            "gt": img_gt,
            "emoca": img_3dmm,            
            "img_name": item["img_name"]
        }
        
        

    def __len__(self) -> int:
        return len(self.data)
