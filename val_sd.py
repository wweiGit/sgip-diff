import os
from argparse import ArgumentParser
import copy

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Mapping, Any, Tuple, Callable, Dict, Literal
from sgip.model import ControlLDM, SwinIR, Diffusion
from sgip.utils.common import instantiate_from_config, to, log_txt_as_img
from sgip.sampler import SpacedSampler, DDIMSampler, EDMSampler
from sgip.model import CLIPTextModelWrapper, prompt_decoder, uncond_prompt_decoder
from sgip.utils.common import wavelet_reconstruction
from PIL import Image
from transformers import CLIPTokenizer

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    # load pre-trained SD weight
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    # load weight
    weight = torch.load(args.pth, map_location="cpu")

    cldm.load_wights_from_ckpt(weight)
    
    cldm.eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained("/root/autodl-tmp/project/SGIP-dev-v2/weights", subfolder="tokenizer")
    text_encoder = CLIPTextModelWrapper.from_pretrained("/root/autodl-tmp/project/SGIP-dev-v2/weights", subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.eval().to(device)
    
    # 离线加载或者在线推理预处理模块
    if not cfg.train.load_clean:
        swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
        sd = torch.load(cfg.train.swinir_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        swinir.load_state_dict(sd, strict=True)
        for p in swinir.parameters():
            p.requires_grad = False
        if accelerator.is_main_process:
            print(f"load SwinIR from {cfg.train.swinir_path}")
        swinir.eval().to(device)
    else:
        print("Using offline clean images instaed of Swinir!")

    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    
    batch_transform = instantiate_from_config(cfg.batch_transform)
    cldm, text_encoder, val_loader = accelerator.prepare(cldm, text_encoder, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)

    # sampler
    if args.sampler=="spaced":
        sampler = SpacedSampler(
            diffusion.betas, diffusion.parameterization, rescale_cfg=False
        )
    elif args.sampler=="ddim":
        sampler = DDIMSampler(
            diffusion.betas, diffusion.parameterization, rescale_cfg=False, eta=0
        )
    else:
        sampler = EDMSampler(
            diffusion.betas, 
            diffusion.parameterization, 
            rescale_cfg=False, 
            solver_type=args.sampler,
            s_churn=0,
            s_tmin=0,
            s_tmax=300,
            s_noise=1,
            eta=1,
            order=1
        )
    
    #pbar = tqdm(
    #    iterable=None,
    #    disable=not accelerator.is_main_process,
    #    unit="batch",
    #    total=len(val_loader),
    #    )
    cldm.eval()
    for i, val_batch in enumerate(val_loader):
        val_batch = to(val_batch, device)
        val_batch = batch_transform(val_batch)
        _, val_lq, val_img_clean, val_text, val_face_id_embed, val_emoca = \
        val_batch["gt"], val_batch["lq"], val_batch["clean"], val_batch["text"], val_batch["face_id_embed"], val_batch["emoca"]
        val_emoca = rearrange(val_emoca, "b h w c -> b c h w").contiguous().float()
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
        
        with torch.no_grad():
            if cfg.train.load_clean:
                val_clean = rearrange(val_img_clean, "b h w c -> b c h w").contiguous().float()
            else:
                val_clean = swinir(val_lq)
            val_cond = pure_cldm.prepare_condition(val_clean)  
            val_cond["c_txt"] = prompt_decoder(val_text, val_face_id_embed, tokenizer, text_encoder, device)
            val_cond["face_id_embed"] = val_face_id_embed
            val_cond["emoca"] = val_emoca
            
            if args.neg_prompt:
                #val_uncond = pure_cldm.prepare_uncondition(val_clean, [args.neg_prompt] * val_clean.shape[0]) 
                val_uncond = {} 
                val_uncond["c_img"] = val_cond["c_img"].detach().clone()
                val_uncond["c_txt"] = uncond_prompt_decoder([args.neg_prompt] * val_clean.shape[0], val_face_id_embed, tokenizer, text_encoder, device)
                val_uncond["face_id_embed"] = val_cond["face_id_embed"].detach().clone()
                val_uncond["emoca"] = val_cond["emoca"].detach().clone()
            else:
                val_uncond = None
            
            bs = val_cond["c_img"].shape[0]
    
            # start_point
            if args.start_point_type == "cond":
                x_T = diffusion.q_sample(
                    val_cond["c_img"],
                    torch.full(
                        (bs,),
                        diffusion.num_timesteps - 1,
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.randn(bs, dtype=torch.float32, device=device),
                )
            else:
                x_T = torch.randn(val_cond["c_img"].shape, dtype=torch.float32, device=device)

            # Noise augmentation
            if args.noise_aug > 0:
                val_cond["c_img"] = diffusion.q_sample(
                    x_start=val_cond["c_img"],
                    t=torch.full(size=(bs,), fill_value=args.noise_aug, device=device),
                    noise=torch.randn_like(val_cond["c_img"]),
                )

            # cond_fn什么作用
            #if args.guidance:
            #    cond_fn.load_target(cond_img * 2 - 1)
                
            sample = sampler.sample(
                model=cldm,
                device=device,
                steps=50,
                x_size=val_cond["c_img"].shape,
                cond = val_cond,
                uncond = val_uncond,
                cfg_scale = args.cfg_scale, 
                x_T = x_T,
                progress=accelerator.is_main_process,
            )
            
            sample = (pure_cldm.vae_decode(sample) + 1) / 2
            
            val_pre = (
                (sample * 255.0)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
                )
            
            os.makedirs(os.path.join(args.save, "ori"), exist_ok=True)
            for j, img in enumerate(val_pre):
                img_save_path = os.path.join(args.save, "ori", val_batch["img_name"][j])
                Image.fromarray(img).save(img_save_path)
            
            
            if args.wave:
                os.makedirs(os.path.join(args.save, "wave"), exist_ok=True)
                
                sample_wave = wavelet_reconstruction(sample, val_clean)
                val_pre_wave = (
                    (sample_wave * 255.0)
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    .cpu()
                    .numpy()
                    )
                
                for k,img_wave in enumerate(val_pre_wave):
                    img_save_path_wave = os.path.join(args.save, "wave", val_batch["img_name"][k])
                    Image.fromarray(img_wave).save(img_save_path_wave)
            

        #pbar.update(1)

    #pbar.close()
    if accelerator.is_main_process:
        print("done!")
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pth", type=str, default="")
    parser.add_argument("--save", type=str, default="/root/autodl-tmp/project/SGIP-dev-v2/work_dirs/inf")
    parser.add_argument("--sampler", type=str, default="spaced")
    parser.add_argument("--neg_prompt", type=str, default = "low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--wave", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=2.0)

    parser.add_argument("--noise_aug", type=int, default=0)
    parser.add_argument("--start_point_type", type=str, default="noise")
    
    args = parser.parse_args()
    main(args)

# low quality, blurry, low-resolution, noisy, unsharp, weird textures