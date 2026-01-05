import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE, NUM_WORKERS, EPOCHS, BASE_LR, MIN_LR, BETAS, WEIGHT_DECAY,
    NUM_CYCLES, MASK_START, MASK_END, GAUSS_NOISE_STD, LOG_INTERVAL, CKPT_DIR
)
from src.datasets.lowlight import PairedLowLightDataset, UnpairedLowLightDataset
from src.models.retinexformer import SimpleRetinexFormer
from src.models.nafnet import SimpleNAFNet
from src.utils.ops import stochastic_mask, add_gaussian_noise, shuffle_residual

def cosine_lr(step, total_steps, base_lr, min_lr):
    if total_steps <= 1:
        return base_lr
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))

def train_arcnet(
    paired_root,
    unpaired_root,
    image_size=(256, 256),
    device="cuda",
):
    # Datasets
    ds_paired = PairedLowLightDataset(paired_root, size=image_size)
    ds_unpaired = UnpairedLowLightDataset(unpaired_root, size=image_size)

    dl_paired = DataLoader(ds_paired, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_unpaired = DataLoader(ds_unpaired, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    # Models
    ire = SimpleRetinexFormer().to(device)
    arm = SimpleNAFNet().to(device)

    # Loss
    l1 = nn.L1Loss()

    # Optimizer
    optimizer = optim.Adam(arm.parameters(), lr=BASE_LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * len(dl_paired)

    step = 0
    for epoch in range(EPOCHS):
        ire.eval()  # IRE is fixed (pretrained synthetic). Replace with your pretrained weights.
        arm.train()

        # Linear schedule for mask ratio
        mask_ratio = MASK_START + (MASK_END - MASK_START) * (epoch / max(1, EPOCHS - 1))

        paired_iter = iter(dl_paired)
        unpaired_iter = iter(dl_unpaired)

        pbar = tqdm(range(len(dl_paired)), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for _ in pbar:
            try:
                lp, gt = next(paired_iter)
            except StopIteration:
                paired_iter = iter(dl_paired)
                lp, gt = next(paired_iter)

            try:
                lr = next(unpaired_iter)
            except StopIteration:
                unpaired_iter = iter(dl_unpaired)
                lr = next(unpaired_iter)

            lp = lp.to(device)
            gt = gt.to(device)
            lr = lr.to(device)

            # Synthetic branch (paired) with occlusion + noise
            with torch.no_grad():
                phi_lp = ire(lp)
            phi_lp_noisy = add_gaussian_noise(phi_lp, GAUSS_NOISE_STD)
            mask_syn = stochastic_mask(phi_lp_noisy, mask_ratio)
            syn_in = mask_syn * phi_lp_noisy

            # Real branch (unpaired) with occlusion
            with torch.no_grad():
                phi_lr = ire(lr)
            mask_real = stochastic_mask(phi_lr, mask_ratio)
            real_in = mask_real * phi_lr

            # Forward ARM on both
            out_syn = arm(syn_in)
            out_real = arm(real_in)

            # Loss on paired synthetic (supervised L1)
            loss_syn = l1(out_syn, gt)

            # Extract degradation residual dk (real branch)
            dk = torch.abs(out_real.detach() - real_in.detach())

            # Build proxy sample xk: inject shuffled residual into synthetic clean
            dk_shuf = shuffle_residual(dk)
            xk = phi_lp_noisy + dk_shuf  # dynamic proxy domain

            # Train ARM again on proxy (matches clean gt)
            out_proxy = arm(xk)
            loss_proxy = l1(out_proxy, gt)

            loss = loss_syn + loss_proxy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Cosine LR
            step += 1
            lr_now = cosine_lr(step, total_steps, BASE_LR, MIN_LR)
            for g in optimizer.param_groups:
                g['lr'] = lr_now

            if step % LOG_INTERVAL == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mask": f"{mask_ratio:.2f}",
                    "lr": f"{lr_now:.2e}"
                })

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"arcnet_epoch{epoch+1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "arm": arm.state_dict(),
            "ire": ire.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        print(f"Saved {ckpt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired_root", type=str, required=True, help="Path to paired dataset root (with low/ and gt/).")
    parser.add_argument("--unpaired_root", type=str, required=True, help="Path to unpaired dataset root (with real/).")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_arcnet(
        paired_root=args.paired_root,
        unpaired_root=args.unpaired_root,
        image_size=tuple(args.size),
        device=args.device,
    )