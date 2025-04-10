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
import itertools
from sgip.model import ControlLDM, SwinIR, Diffusion
from sgip.utils.common import instantiate_from_config, to, log_txt_as_img
from sgip.sampler import SpacedSampler
from sgip.model import CLIPTextModelWrapper, prompt_decoder
import datetime
from PIL import Image
from transformers import CLIPTokenizer
from sgip.utils.common import wavelet_reconstruction

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(666, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    if not args.save_dir:
        args.save_dir = cfg.train.exp_dir
        
    os.makedirs(args.save_dir, exist_ok=True)
    # Setup an experiment folder:
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(args.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {args.save_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    tokenizer = CLIPTokenizer.from_pretrained("/root/autodl-tmp/project/SGIP-dev-v2/weights", subfolder="tokenizer")
    text_encoder = CLIPTextModelWrapper.from_pretrained("/root/autodl-tmp/project/SGIP-dev-v2/weights", subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

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

    # Setup optimizer:
    params_to_opt = itertools.chain(cldm.id_proj.parameters(), 
                                    cldm.text_proj.parameters(), 
                                    cldm.controlnet_cond_embedding.parameters(), 
                                    cldm.controlnet.parameters())
    
    opt = torch.optim.AdamW(params_to_opt, lr=cfg.train.learning_rate)

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    
    if accelerator.is_main_process:
        print(f"Dataset contains {len(dataset):,} images")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training:
    cldm.train().to(device)
    diffusion.to(device)
    cldm, opt, loader, val_loader = accelerator.prepare(cldm, opt, loader, val_loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    noise_aug_timestep = cfg.train.noise_aug_timestep

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    
    # sampler
    sampler = SpacedSampler(
        diffusion.betas, diffusion.parameterization, rescale_cfg=False
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(args.save_dir)
        print(f"Training for {max_steps} steps...")

    log_file = os.path.join(args.save_dir, "log.txt")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            batch = to(batch, device)
            batch = batch_transform(batch)
            
            #  "lq" "gt" "clean" "emoca" "face_id_embed" "img_name" "text"
            
            gt, lq, img_clean, text, face_id_embed, emoca = batch["gt"], batch["lq"], batch["clean"], batch["text"], batch["face_id_embed"], batch["emoca"]
            
            #print("input:", gt.shape, lq.shape)
            #print("input:", img_clean.shape, text)
            #print("input:", face_id_embed.shape, emoca.shape)
            
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
            emoca = rearrange(emoca, "b h w c -> b c h w").contiguous().float()
            
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                ################ 离线加载clean替代swinir推理 ################
                if cfg.train.load_clean:
                    clean = rearrange(img_clean, "b h w c -> b c h w").contiguous().float()
                else:
                    clean = swinir(lq)
                ###########################################################
                cond = pure_cldm.prepare_condition(clean)
                cond["c_txt"] = prompt_decoder(text, face_id_embed, tokenizer, text_encoder, device)
                cond["face_id_embed"] = face_id_embed
                cond["emoca"] = emoca
                # noise augmentation
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

                p_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # 打开文件进行写入
                with open(log_file, 'a') as f:  # 'a'模式表示附加写入，不会覆盖文件内容
                    f.write("Time:{}, Steps:{}, Epoch:{}, avg_loss:{} \n".format(p_time, global_step, epoch, avg_loss))

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    controlnet_ckpt = pure_cldm.controlnet.state_dict()
                    id_proj_ckpt = pure_cldm.id_proj.state_dict()
                    text_proj_ckpt = pure_cldm.text_proj.state_dict()
                    controlnet_cond_embedding_ckpt = pure_cldm.controlnet_cond_embedding.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save({"controlnet": controlnet_ckpt, "id_proj": id_proj_ckpt, 
                                "text_proj": text_proj_ckpt, "controlnet_cond_embedding": 
                                    controlnet_cond_embedding_ckpt}, ckpt_path)

            ############################### 开发推理代码0310 ####################################
            if global_step % cfg.train.val_every == 0 or global_step == max_steps:
                cldm.eval()
                save_path = os.path.join(args.save_dir, "results", str(global_step))
                os.makedirs(save_path, exist_ok=True)
                
                with torch.no_grad():
                    for i, val_batch in enumerate(val_loader):
                        val_batch = to(val_batch, device)
                        val_batch = batch_transform(val_batch)
                        _, val_lq, val_img_clean, val_text, val_face_id_embed, val_emoca = \
                        val_batch["gt"], val_batch["lq"], val_batch["clean"], val_batch["text"], val_batch["face_id_embed"], val_batch["emoca"]
                        val_emoca = rearrange(val_emoca, "b h w c -> b c h w").contiguous().float()
                        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
                        
                        if cfg.train.load_clean:
                            val_clean = rearrange(val_img_clean, "b h w c -> b c h w").contiguous().float()
                        else:
                            val_clean = swinir(val_lq)
                        val_cond = pure_cldm.prepare_condition(val_clean)  
                        val_cond["c_txt"] = prompt_decoder(val_text, val_face_id_embed, tokenizer, text_encoder, device)
                        val_cond["face_id_embed"] = val_face_id_embed
                        val_cond["emoca"] = val_emoca

                        sample = sampler.sample(
                            model=cldm,
                            device=device,
                            steps=50,
                            x_size=val_cond["c_img"].shape,
                            cond=val_cond,
                            uncond=None,
                            cfg_scale=1.0,
                            progress=accelerator.is_main_process,
                        )
                        
                        sample = (pure_cldm.vae_decode(sample) + 1) / 2
                    
                    
                        sample = wavelet_reconstruction(sample, val_clean)
                        val_pre = (
                            (sample * 255.0)
                            .clamp(0, 255)
                            .to(torch.uint8)
                            .permute(0, 2, 3, 1)
                            .contiguous()
                            .cpu()
                            .numpy()
                            )
                        for j,sample in enumerate(val_pre):
                            img_save_path = os.path.join(save_path, val_batch["img_name"][j])
                            Image.fromarray(sample).save(img_save_path)
                        
                        if i >= 20 and global_step < max_steps:
                            break
                        
                cldm.train()
            accelerator.wait_for_everyone()
            
            ###################################################################

            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)
            

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="")
    args = parser.parse_args()

    main(args)

