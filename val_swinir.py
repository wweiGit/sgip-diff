import os
from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from einops import rearrange
from sgip.model import SwinIR
from sgip.utils.common import instantiate_from_config,to
import numpy as np
import cv2
from tqdm import tqdm

def save_img_ori(output, out_path):
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(out_path), output)
    
def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(args.pth, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False

    # Setup data:
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
    )
    swinir.eval().to(device)

    os.makedirs(args.save_dir, exist_ok=True)
        
    for batch in tqdm(val_loader, desc="Processing batches", unit="batch"):
        lq= batch['lq'].to(device)
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
        val_pred = swinir(lq)

        ##########遍历 batch 内的每张图像并保存############
        for i in range(len(val_pred)):
            save_img_ori(val_pred[i], os.path.join(args.save_dir, batch['img_name'][i]))
        #################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pth", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='/root/autodl-tmp/project/SGIP-dev/inf_results/stage1_expos_train')
    args = parser.parse_args()
    main(args)
