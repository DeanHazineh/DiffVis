import os
import itertools  # NEVER USE ITERTOOLS.CYCLE ON TRAINING DATA WITH RANDOM AUGMENTATIONS
import shutil
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from dflat.render import hsi_to_rgb
from skimage.metrics import structural_similarity as ssim
from diffvis.group_transforms import reverse_transform, permute_dimensions
from torch.cuda.amp import autocast, GradScaler


def make_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return


def empty_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))


# Compute the Exponential Moving Average (EMA)
def compute_ema(data, alpha=0.1):
    ema = [data[0]]  # Start with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[i - 1])
    return np.array(ema)


def batch_ssim(batch_target, batch_ref, data_range=1.0):
    ssim_values = []
    for i in range(batch_target.shape[0]):
        current_ssim = ssim(
            batch_target[i], batch_ref[i], data_range=data_range, channel_axis=-1
        )
        ssim_values.append(current_ssim)
    return np.array(ssim_values)


def batch_psnr(target, ref):
    mse = np.mean((target - ref) ** 2, axis=(-1, -2, -3))
    return mse, 20 * np.log10(1 / np.sqrt(mse))


def load_checkpoint_with_mismatches(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract the state_dict from the checkpoint
    checkpoint_state_dict = checkpoint["state_dict"]

    # Get the current model state_dict
    model_state_dict = model.state_dict()

    # Prepare a new state_dict for the model
    new_state_dict = {}

    for name, param in model_state_dict.items():
        if (
            name in checkpoint_state_dict
            and param.size() == checkpoint_state_dict[name].size()
        ):
            # If sizes match, copy the parameter from the checkpoint
            new_state_dict[name] = checkpoint_state_dict[name]
        else:
            # If sizes mismatch, initialize randomly
            # Here we assume parameters are initialized from a normal distribution for weights
            # and zeros for biases, this can be adjusted as necessary
            if "weight" in name:
                new_state_dict[name] = torch.randn(param.size())
            elif "bias" in name:
                new_state_dict[name] = torch.zeros(param.size())


def custom_collate_fn(batch):
    collated_batch = {}

    for key in batch[0]:
        if torch.is_tensor(batch[0][key]):
            tensors = []
            for item in batch:
                if len(item[key].shape) == 3:  # If the shape is [C, H, W]
                    tensors.append(item[key].unsqueeze(0))  # Add batch dimension
                else:
                    tensors.append(item[key])
            collated_batch[key] = torch.cat(tensors, dim=0)
        else:
            concatenated_list = []
            for item in batch:
                if isinstance(item[key], str):
                    concatenated_list.append(item[key])
                elif isinstance(item[key], list):
                    concatenated_list.extend(item[key])
                else:
                    raise ValueError(f"Unsupported data type for item[{key}]")
            collated_batch[key] = concatenated_list

    return collated_batch


def step_scheduler(scheduler, epoch, original_T_max):
    if epoch >= original_T_max:
        for param_group in scheduler.optimizer.param_groups:
            param_group["lr"] = scheduler.eta_min
    else:
        scheduler.step(epoch)
    return


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        ckpt_path,
        batch_size,
        max_steps,  # In this version this maps to epochs
        lr,
        gradient_accumulation_steps,
        snapshot_every_n,
        sample_img_size,
        disp_num_samples,
        save_intermediate_ckpt,
        start_clean=False,
        skip_params=[],
        valid_dataset=None,
        train_valid_split=0.85,
        dl_workers=None,
        dl_pin_mem=True,
        skip_valid_step=False,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.0,
        load_optimizer=True,
        exp_decay_lr=1e-6,
        alpha=0.9999,
        valid_every_n=1,
        hard_reset_lr=False,
    ):
        self.model = model.to("cuda")
        self.input_key = model.input_key
        self.xcond_key = model.xcond_key
        self.ccond_key = model.ccond_key
        # mask = model.mask
        # self.mask = mask.cpu().numpy() if mask is not None else None
        self.mask = None  # Unused

        if valid_dataset is None:
            torch.manual_seed(42)
            train_size = int(train_valid_split * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = random_split(
                train_dataset, [train_size, valid_size]
            )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=dl_workers,
            pin_memory=dl_pin_mem,
            collate_fn=custom_collate_fn,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=dl_pin_mem,
            collate_fn=custom_collate_fn,
        )
        self.valid_every_n = valid_every_n
        self.ckpt_path = ckpt_path
        self.max_epochs = max_steps
        self.lr = lr
        self.gaccum_steps = gradient_accumulation_steps
        self.snapshot_every_n = snapshot_every_n
        self.sample_img_size = sample_img_size
        self.save_intermediate_ckpt = save_intermediate_ckpt
        self.start_clean = start_clean
        self.skip_params = skip_params
        self.skip_valid_step = skip_valid_step
        self.snum = batch_size if disp_num_samples > batch_size else disp_num_samples
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.load_optimizer = load_optimizer
        self.exp_decay_lr = exp_decay_lr
        self.__init_dir()
        self.alpha = alpha
        self.hard_reset_lr = hard_reset_lr

    def __init_dir(self, overwrite=False):
        ckpt_path = self.ckpt_path
        directories = [
            ckpt_path,
            os.path.join(ckpt_path, "model_snapshots/"),
            os.path.join(ckpt_path, "train_logs/"),
        ]
        for directory in directories:
            if os.path.exists(directory) and overwrite:
                empty_directory(directory)
            make_directory_if_not_exists(directory)

        return

    def plot_drawn_samples(self, batch, fname):
        return None

    def fit(self):
        model = self.model
        train_dl = self.train_dataloader
        valid_dl = self.valid_dataloader
        valid_iter = itertools.cycle(valid_dl)
        train_iter = itertools.cycle(train_dl)  # Used only for visualization data

        ckpt_folder = self.ckpt_path + "model_snapshots/"
        log_folder = self.ckpt_path + "train_logs/"
        last_ckpt_path = ckpt_folder + "ckpt_last.ckpt"
        best_ckpt_path = ckpt_folder + "best_ckpt.ckpt"
        if self.start_clean:
            self.__init_dir(overwrite=True)

        # Fix grad accumulation step number as precaution
        dl_len = len(train_dl)
        gaccum_steps = dl_len if self.gaccum_steps > dl_len else self.gaccum_steps
        print(f"Gradient accumulation steps: {gaccum_steps}")

        # Set up optimizer
        model_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad or any(
                name.startswith(key) for key in self.skip_params
            ):
                continue
            else:
                if name.startswith("model."):
                    model_params.append(param)

        print("Learning Rate: ", self.lr)
        optimizer = AdamW(
            model_params,
            self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.exp_decay_lr,
        )

        # Load the last checkpoint to resume paused training
        if os.path.exists(last_ckpt_path):
            checkpoint = torch.load(
                last_ckpt_path, map_location="cpu", weights_only=False
            )
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict, strict=True)
            model.to("cuda")

            optimizer_ckpt = checkpoint["optimizer_state_dict"]
            start_epoch = checkpoint["start_epoch"]
            train_losses = checkpoint["train_losses"]
            test_losses = checkpoint["test_losses"]
            train_metrics = checkpoint["train_metrics"]
            test_metrics = checkpoint["test_metrics"]
            epoch_vec = checkpoint["epoch_vec"]

            if self.load_optimizer:
                optimizer.load_state_dict(optimizer_ckpt)
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                alpha = checkpoint["alpha"]
                ema_grads = checkpoint["ema_grads"]
                ema_grads = {
                    key: tensor.to("cuda") for key, tensor in ema_grads.items()
                }
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.max_epochs,
                    eta_min=self.exp_decay_lr,
                )
                alpha = self.alpha
                ema_grads = {
                    name: torch.zeros_like(param).to("cuda")
                    for name, param in model.named_parameters()
                }
            print(f"Loaded checkpoint from epoch {start_epoch}")

        else:
            model.to("cuda")
            start_epoch = 0
            train_losses = []
            test_losses = []
            train_metrics = []
            test_metrics = []
            epoch_vec = []
            ema_grads = {
                name: torch.zeros_like(param)
                for name, param in model.named_parameters()
            }
            alpha = self.alpha
        original_T_max = scheduler.state_dict()["T_max"]
        print("original t max: ", original_T_max)

        if os.path.exists(best_ckpt_path):
            best_state = torch.load(best_ckpt_path, weights_only=False)
        else:
            best_state = {}

        # Run Training
        for epoch in range(start_epoch, self.max_epochs + 1):
            start_time = time.time()
            model.train()
            epoch_loss = 0

            for batch in train_dl:
                optimizer.zero_grad(
                    set_to_none=True
                )  # Clears accumulated gradients efficiently
                loss = model.training_step(batch)
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        ema_grads[name] = (
                            alpha * param.grad + (1 - alpha) * ema_grads[name]
                        )
                        param.grad = ema_grads[name]

                optimizer.step()
                torch.cuda.empty_cache()  # Releases cached memory back to the system
                gc.collect()  # Runs garbage collection to free unused tensors

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(train_dl)
            train_losses.append(epoch_loss)
            if self.hard_reset_lr:
                scheduler.step()
            else:
                step_scheduler(scheduler, epoch, original_T_max)

            # Valid epoch
            if not self.skip_valid_step and epoch % self.valid_every_n == 0:
                model.eval()
                epoch_loss = 0
                for batch in valid_dl:
                    with torch.no_grad():
                        loss = model.training_step(batch)
                        epoch_loss += loss.item()
                epoch_loss = epoch_loss / len(valid_dl)
                test_losses.append(epoch_loss)
            else:
                test_losses.append(np.nan)
            end_time = time.time()
            current_lr = [optimizer.param_groups[0]["lr"]]
            time_elapsed = end_time - start_time
            print(
                f"Epoch {epoch+1} Finished in {time_elapsed:.2f}| lr: {current_lr} Train_loss: {train_losses[-1]:.2e} Test_Loss: {test_losses[-1]:.2e}"
            )

            if not np.isnan(test_losses[-1]):
                filt = [x for x in test_losses[:-1] if not np.isnan(x)]
                prev_min = np.min(filt) if filt else test_losses[-1]
                if test_losses[-1] <= prev_min:
                    print(f"New Best Loss {test_losses[-1]} vs {prev_min}")
                    best_state = {
                        "start_epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_losses": train_losses,
                        "test_losses": test_losses,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "epoch_vec": epoch_vec,
                        "ema_grads": ema_grads,
                        "alpha": alpha,
                    }

            if epoch % self.snapshot_every_n == 0 or epoch == self.max_epochs:
                # Save a loss figure
                ema_train = compute_ema(train_losses, alpha=0.10)
                # ema_test = compute_ema(test_losses, alpha=0.10)
                ema_test = test_losses
                plt.figure(figsize=(10, 5))
                ax = plt.gca()
                ax.plot(train_losses, ".", color="gray", label="train", alpha=0.1)
                ax.plot(ema_train, "r-", label="train", linewidth=2, alpha=0.7)
                ax.plot(ema_test, "g-", label="valid", linewidth=2, alpha=0.7)
                ax.set_xlabel("batch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                lines1, labels1 = ax.get_legend_handles_labels()
                ax.legend(lines1, labels1, loc=0)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.tight_layout()
                plt.savefig(log_folder + "training_loss.png")
                plt.close()

                # Draw some samples for visualization
                out = self.plot_drawn_samples(
                    next(train_iter), f"epoch_{epoch}_train.png"
                )
                train_metrics.append(out)
                out = self.plot_drawn_samples(
                    next(valid_iter), f"epoch_{epoch}_test.png"
                )
                test_metrics.append(out)
                epoch_vec.append(epoch)

                state = {
                    "start_epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "epoch_vec": epoch_vec,
                    "ema_grads": ema_grads,
                    "alpha": alpha,
                }
                torch.save(state, last_ckpt_path)
                print("Saved Model Checkpoint!")

                if best_state:
                    torch.save(best_state, best_ckpt_path)

                train_met = np.stack(train_metrics)
                test_met = np.stack(test_metrics)
                labels = ["SSIM", "MSE", "PSNR"]
                markers = ["o", "x", "v"]
                fig, ax = plt.subplots(1, 3, figsize=(9, 2))
                for li, lab in enumerate(labels):
                    ax[li].plot(
                        train_met[:, li],
                        color="red",
                        marker=markers[li],
                        label="train " + lab,
                    )
                    ax[li].plot(
                        # epoch_vec,
                        test_met[:, li],
                        color="g",
                        marker=markers[li],
                        label="test " + lab,
                    )
                    ax[li].legend()
                    ax[li].grid(True, which="both", linestyle="--", linewidth=0.5)
                plt.tight_layout()
                plt.savefig(log_folder + "training_metrics.png")
                plt.close()

                time.sleep(1)

        return


class Trainer_HSI(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask = self.model.mask

    def plot_drawn_samples(self, batch, fname):
        snum = self.snum
        input, xcond, ccond = self.model.get_input(batch)
        snum = np.minimum(snum, input.shape[0])

        input = input[:snum]
        xcond = xcond[:snum] if xcond is not None else None
        ccond = ccond[:snum] if ccond is not None else None

        samples, _ = self.model.sample(
            self.sample_img_size,
            batch_size=snum,
            variance_n=1,
            return_intermediate=False,
            x_start=None,
            xcond=xcond,
            ccond=ccond,
        )

        samples = samples[
            -1, :, 0, :, :, :
        ]  # Pull from [t, b, v, c, h, w] to [b, c, h, w]
        hsi_est = reverse_transform(samples)
        hsi_gt = reverse_transform(input).astype(np.float32)

        if self.mask is None:
            mask = np.ones((1, *hsi_est.shape[1:3], 1))
        else:
            mask = self.mask[None, :, :, None]
        ssim_imgs = batch_ssim(hsi_est * mask, hsi_gt * mask)
        mse, psnr = batch_psnr(hsi_est * mask, hsi_gt * mask)

        # Plot the images in RGB space
        bs, chs = samples.shape[0], samples.shape[1]
        numl = 8
        lam = np.linspace(400e-9, 700e-9, chs)
        # lam = np.linspace(450e-9, 650e-9, chs)

        wlidx = np.linspace(0, chs - 1, numl).astype(int)
        rgb_samples = hsi_to_rgb(hsi_est, lam)
        rgb_target = hsi_to_rgb(hsi_gt, lam)
        fig, ax = plt.subplots(bs, numl + 2, figsize=(3 * (numl + 2), 3 * bs))
        for b in range(bs):
            ax[b, 0].imshow(rgb_target[b])
            ax[b, 1].imshow(rgb_samples[b])
            for ai, wi in enumerate(wlidx):
                ax[b, ai + 2].imshow(hsi_est[b, :, :, wi], vmin=0, vmax=1, cmap="gray")

        ax[0, 0].set_title("AIF")
        ax[0, 1].set_title("Rec.")
        for ai, wi in enumerate(wlidx):
            ax[0, ai + 2].set_title(f"{int(lam[wi]*1e9)}")

        for thisax in ax.flatten():
            thisax.axis("off")
        plt.tight_layout()
        plt.savefig(self.ckpt_path + f"train_logs/{fname}")
        plt.close()

        return np.mean(ssim_imgs), np.mean(mse), np.mean(psnr)
