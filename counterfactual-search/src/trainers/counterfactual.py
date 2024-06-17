from copy import deepcopy
from unittest import loader

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.utils import save_image
from tqdm import tqdm

from src.visualizations import confmat_vis_img
from src.datasets import get_dataloaders
from src.datasets.augmentations import get_transforms
from src.models.cgan import CounterfactualCGAN
from src.models.classifier import compute_sampler_condition_labels, predict_probs
from src.trainers.trainer import BaseTrainer
from src.utils.avg_meter import AvgMeter
from src.utils.generic_utils import save_model

import os
import time
import copy
from random import shuffle

import torch
from torchvision.utils import save_image
from tqdm import tqdm
import os


import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
#import tqdm.notebook as tqdm

import matplotlib.pyplot as plt
# %matplotlib inline

imagenet_mean = 0.485, 0.456, 0.406
imagenet_std = 0.229, 0.224, 0.225
grayscale_coefs = 0.2989, 0.587, 0.114

grayscale_mean = sum(m * c for m, c in zip(imagenet_mean, grayscale_coefs))
grayscale_std = sum(m * c for m, c in zip(imagenet_std, grayscale_coefs))

# def imshow(inp, size=(30, 30), title=None, ax=None):
#     inp = inp * grayscale_std + grayscale_mean
#     inp = inp.numpy().transpose((1, 2, 0))
#     if ax is None:
#         plt.figure(figsize=size)
#         plt.imshow(inp, ax=ax)
#     ax.imshow(inp)
#     if title is not None:
#         plt.title(title, size=30)

def save_imagee(inp, filename):
    inp = inp * grayscale_std + grayscale_mean  # unnormalize
    inp = inp.numpy().transpose((1, 2, 0))  # Convert from Tensor image
    plt.imshow(inp)
    plt.axis('off')  # Hide axes
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def imshow(inp, size=(30, 30), title=None, filename=None, ax=None):
    inp = inp * grayscale_std + grayscale_mean  # unnormalize
    inp = inp.numpy().transpose((1, 2, 0))  # Convert from Tensor image
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    ax.imshow(inp)
    ax.axis('off')  # Hide axes
    if title is not None:
        ax.set_title(title, size=30)
    if filename is not None:
        save_image(inp, filename)

        
# def imshow(inp, size=(30, 30), title=None, ax=None):
#     inp = inp * grayscale_std + grayscale_mean  # unnormalize
#     inp = inp.numpy().transpose((1, 2, 0))  # Convert from Tensor image
#     if ax is None:
#         fig, ax = plt.subplots(figsize=size)
#     ax.imshow(inp)
#     ax.axis('off')  # Hide axes
#     if title is not None:
#         ax.set_title(title, size=30)
#     plt.show()  # Show the plot


class CounterfactualTrainer(BaseTrainer):
    def __init__(self, opt: edict, model: CounterfactualCGAN, continue_path: str = None) -> None:
        super().__init__(opt, model, continue_path)
        self.cf_vis_dir_train = self.logging_dir / 'counterfactuals/train'
        self.cf_vis_dir_train.mkdir(exist_ok=True, parents=True)
        
        self.cf_vis_dir_val = self.logging_dir / 'counterfactuals/val'
        self.cf_vis_dir_val.mkdir(exist_ok=True)

        self.compute_norms = self.opt.get('compute_norms', False)
        # 64, 192, 768, 2048
        self.fid_features = opt.get('fid_features', 768)
        self.val_fid = FrechetInceptionDistance(self.fid_features, normalize=True).to(self.device)
        
        self.cf_gt_seg_mask_idx = opt.get('cf_gt_seg_mask_idx', -1)
        self.cf_threshold = opt.get('cf_threshold', 0.25)
        self.val_iou_xc = BinaryJaccardIndex(self.cf_threshold).to(self.device)
        self.val_iou_xfx = BinaryJaccardIndex(self.cf_threshold).to(self.device)

    def restore_state(self):
        latest_ckpt = max(self.ckpt_dir.glob('*.pth'), key=lambda p: int(p.name.replace('.pth', '').split('_')[1]))
        load_ckpt = latest_ckpt if self.ckpt_name is None else (self.ckpt_dir / self.ckpt_name)
        state = torch.load(load_ckpt)
        self.batches_done = state['step']
        self.current_epoch = state['epoch']
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer_G.load_state_dict(state['optimizers'][0])
        self.model.optimizer_D.load_state_dict(state['optimizers'][1])
        self.logger.info(f"Restored checkpoint {load_ckpt} ({state['date']})")

    def save_state(self) -> str:
        return save_model(self.opt, self.model, (self.model.optimizer_G, self.model.optimizer_D), self.batches_done, self.current_epoch, self.ckpt_dir)

    def get_dataloaders(self, ret_cf_labels:bool=False, skip_cf_sampler=True) -> tuple:
        transforms = get_transforms(self.opt.dataset)
        if self.opt.task_name == 'counterfactual' and not skip_cf_sampler: # NB
            # compute sampler labels to create batches with uniformly distributed labels
            params = edict(deepcopy(self.opt.dataset), use_sampler=False, shuffle_test=False)
            # GAN's train data is expected to be classifier's validation data
            train_loader, _ = get_dataloaders(params, {'train': transforms['val'], 'val': transforms['train']})
            posterior_probs, _ = predict_probs(train_loader, self.model.classifier_f, task='binary' if self.model.n_classes == 1 else 'multiclass')
            # sampler labels are just digitized classifier's posterior probabilities on input images
            # i.e we sample into batches images where classifier predicts p<0.5 in half of the cases and p>0.5 otherwise
            sampler_labels = compute_sampler_condition_labels(posterior_probs, self.model.explain_class_idx, self.model.num_bins)
            self.logger.info(f'Precomputed condition labels for sampling. Num positive conditions: {sampler_labels.sum()}')
        else:
            sampler_labels = None
        loaders = get_dataloaders(self.opt.dataset, transforms, sampler_labels=sampler_labels)
        print(f"\n++++++++++++++++++++++++++++++IMAGE SAVE++++++++++++++++++\n")
        batchh_train = next(iter(loaders[0]))
        batchh_test = next(iter(loaders[1]))
        
        imagess_train = batchh_train['image'][:9]  # Extract the first 9 images
        imagess_test = batchh_test['image'][:9]  # Extract the first 9 images
        
        outt_train = torchvision.utils.make_grid(imagess_train, nrow=3)
        outt_test = torchvision.utils.make_grid(imagess_test, nrow=3)
        save_imagee(outt_train, "example_image_outt_train.png")  # Save the image to a file
        save_imagee(outt_test, "example_image_outt_test.png")  # Save the image to a file


        
        # # batchh = next(iter(loaders))
        # batchh = next(iter(loaders[0]))
        # outt = torchvision.utils.make_grid(batchh['image'][:9], nrow=3)
        
        # imshow(outt, ax=plt.gca())
        # imshow(outt, ax=plt.gca(), title="Example Image", filename="example_image.png")
        
        return (loaders, sampler_labels) if ret_cf_labels else loaders

    def training_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()

        stats = AvgMeter()
        epoch_steps = self.opt.get('epoch_steps')
        print(f"epoch_steps: {epoch_steps}")
        print(len(loader))
        with tqdm(enumerate(loader), desc=f'Training epoch {self.current_epoch}', leave=False, total=epoch_steps or len(loader)) as prog:
            for i, batch in prog:
                if i == epoch_steps:
                    break
                self.batches_done = self.current_epoch * len(loader) + i
                sample_step = self.batches_done % self.opt.sample_interval == 0
                outs = self.model(batch, training=True, compute_norms=sample_step and self.compute_norms, global_step=self.batches_done)
                stats.update(outs['loss'])
                if sample_step:
                    if self.opt.get('log_visualizations', True):
                        save_image(outs['gen_imgs'][:16].data, self.vis_dir / ('%d_train_%d.jpg' % (self.current_epoch, i)), nrow=4, normalize=True)
                    postf = '[Batch %d/%d] [D loss: %f] [G loss: %f]' % (i, len(loader), outs['loss']['d_loss'], outs['loss']['g_loss'])
                    prog.set_postfix_str(postf, refresh=True)
                    if self.compute_norms:
                        for model_name, norms in self.model.norms.items():
                            self.logger.log(norms, self.batches_done, f'{model_name}_gradients_norm')
        epoch_stats = stats.average()
        if self.current_epoch % self.opt.eval_counter_freq == 0:
            epoch_stats.update(self.evaluate_counterfactual(loader, phase='train'))
        self.logger.log(epoch_stats, self.current_epoch, 'train')
        self.logger.info(
            '[Finished training epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
        )
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()

        stats = AvgMeter()
        avg_pos_to_neg_ratio = 0.0
        for i, batch in tqdm(enumerate(loader), desc=f'Validation epoch {self.current_epoch}', leave=False, total=len(loader)):
            avg_pos_to_neg_ratio += batch['label'].sum() / batch['label'].shape[0]
            outs = self.model(batch, validation=True)
            stats.update(outs['loss'])
            if i % self.opt.sample_interval == 0 and self.opt.get('vis_gen', True):
                save_image(outs['gen_imgs'][:16].data, self.vis_dir / ('%d_val_%d.jpg' % (self.batches_done, i)), nrow=4, normalize=True)
        self.logger.info('[Average positives/negatives ratio in batch: %f]' % round(avg_pos_to_neg_ratio.item() / len(loader), 3))
        epoch_stats = stats.average()
        if self.current_epoch % self.opt.eval_counter_freq == 0:
            epoch_stats.update(self.evaluate_counterfactual(loader, phase='val'))
        self.logger.log(epoch_stats, self.current_epoch, 'val')
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
        )
        return epoch_stats

   
    @torch.no_grad()
    def evaluate_counterfactual(self, loader, phase='val', tau=0.8, skip_fid=False, postprocess_morph=False):
        self.model.eval()

        cf_dir = self.cf_vis_dir_train if phase == 'train' else self.cf_vis_dir_val
        import os
        # Create directories for each type of image
        real_image_dir = cf_dir / 'real_images'
        generated_image_dir = cf_dir / 'generated_images'
        difference_image_dir = cf_dir / 'difference_images'
        all_dir = cf_dir / 'real_gen_diff_images'
        os.makedirs(real_image_dir, exist_ok=True)
        os.makedirs(generated_image_dir, exist_ok=True)
        os.makedirs(difference_image_dir, exist_ok=True)
        os.makedirs(all_dir, exist_ok=True)

        classes = []
        cv_y_true, cv_y_pred = [], []
        posterior_true, posterior_pred = [], []

        pred_num_abnormal_samples = 0
        true_num_abnormal_samples = 0
        for i, batch in tqdm(enumerate(loader), desc='Validating counterfactuals', leave=False, total=len(loader)):
            real_imgs = batch['image'].cuda(non_blocking=True).float()
            cf_gt_masks = batch['masks'][:, self.cf_gt_seg_mask_idx].cuda(non_blocking=True).float()
            labels = batch['label']
            true_num_abnormal_samples += labels.sum()
            B = labels.shape[0]

            self.model: CounterfactualCGAN
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.model.posterior_prob(real_imgs)

            real_neg_pos_group = ((real_f_x < 0.2) | (real_f_x > 0.8)).view(-1)
            if not real_neg_pos_group.any():
                continue

            real_imgs, cf_gt_masks, labels = real_imgs[real_neg_pos_group], cf_gt_masks[real_neg_pos_group], labels[real_neg_pos_group.cpu()]
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = (
                real_f_x[real_neg_pos_group],
                real_f_x_discrete[real_neg_pos_group],
                real_f_x_desired[real_neg_pos_group],
                real_f_x_desired_discrete[real_neg_pos_group],
            )

            cv_y_true.extend(real_f_x_desired_discrete.cpu().squeeze(1).numpy())
            posterior_true.extend(real_f_x.cpu().squeeze(1).numpy())
            classes.extend(labels.cpu().numpy())

            gen_cf_c = self.model.explanation_function(real_imgs, real_f_x_desired_discrete)
            gen_cf_fx = self.model.explanation_function(real_imgs, real_f_x_discrete)

            gen_f_x, gen_f_x_discrete, _, _ = self.model.posterior_prob(gen_cf_c)
            cv_y_pred.extend(gen_f_x_discrete.cpu().squeeze(1).numpy())
            posterior_pred.extend(gen_f_x.cpu().squeeze(1).numpy())

            real_imgs.add_(1).div_(2)
            gen_cf_c.add_(1).div_(2)
            gen_cf_fx.add_(1).div_(2)

            diff = (real_imgs - gen_cf_c).abs()
            diff_seg = (diff > self.cf_threshold).byte()
            diff2 = (gen_cf_fx - gen_cf_c).abs()
            diff2_seg = (diff2 > self.cf_threshold).byte()

            abnormal_mask = labels.bool()
            if abnormal_mask.any():
                pred_num_abnormal_samples += abnormal_mask.sum()
                self.val_iou_xc.update(diff_seg[abnormal_mask].squeeze(1), cf_gt_masks[abnormal_mask].squeeze(1))
                self.val_iou_xfx.update(diff2_seg[abnormal_mask].squeeze(1), cf_gt_masks[abnormal_mask].squeeze(1))

            vis = torch.stack((
                real_imgs[0], torch.zeros_like(real_imgs[0]), torch.zeros_like(real_imgs[0]), 
                gen_cf_c[0], diff[0], diff_seg[0],
                gen_cf_fx[0], diff2[0], diff2_seg[0],
            ), dim=0).permute(0, 2, 3, 1)
            vis = torch.cat((vis, vis, vis), 3)
            vis_confmat = confmat_vis_img(cf_gt_masks[0].unsqueeze(0).unsqueeze(0), diff_seg[0].unsqueeze(0), normalized=True)[0]
            vis[1] = 0.3 * vis[0] + 0.7 * vis_confmat
            vis[2] = vis_confmat
            vis = vis.permute(0, 3, 1, 2)

            vis_path = cf_dir / (f'epoch_{self.current_epoch}_counterfactual_{i}_label_{labels[0]}_true_{real_f_x_desired_discrete[0][0]}_pred_{gen_f_x_discrete[0][0]}.jpg')
            save_image(vis.data, vis_path, nrow=3, normalize=False)

            # Save the real image, generated image, and difference image in separate directories
            real_image_path = real_image_dir / (f'epoch_{self.current_epoch}_real_image_{i}.jpg')
            generated_image_path = generated_image_dir / (f'epoch_{self.current_epoch}_generated_image_{i}.jpg')
            difference_image_path = difference_image_dir / (f'epoch_{self.current_epoch}_difference_image_{i}.jpg')
            
            #all_image_path = all_dir / (f'epoch_{self.current_epoch}_difference_image_{i}.jpg')

            save_image(real_imgs.data, real_image_path, nrow=1, normalize=False)
            save_image(gen_cf_c.data, generated_image_path, nrow=1, normalize=False)
            save_image(diff.data, difference_image_path, nrow=1, normalize=False)

            # Create title tensors with ASCII values
            real_title_ascii = ord('R')
            generated_title_ascii = ord('G')
            difference_title_ascii = ord('D')

            # Create title tensors filled with ASCII values
            real_title = torch.full((1, real_imgs.size(1), real_imgs.size(2), 1), fill_value=real_title_ascii, dtype=torch.uint8)
            generated_title = torch.full((1, real_imgs.size(1), real_imgs.size(2), 1), fill_value=generated_title_ascii, dtype=torch.uint8)
            difference_title = torch.full((1, real_imgs.size(1), real_imgs.size(2), 1), fill_value=difference_title_ascii, dtype=torch.uint8)

            # Move title tensors to the CUDA device
            real_title = real_title.to(real_imgs.device)
            generated_title = generated_title.to(real_imgs.device)
            difference_title = difference_title.to(real_imgs.device)


            grid_image = torch.cat((real_imgs.data, gen_cf_c.data, diff.data), dim=3)
            # Reshape the concatenated tensor to have three columns and three rows
            #grid_image = grid_image.view(grid_image.size(0), -1, 3, grid_image.size(2))




            # Unnormalize the grid image
            grid_image = grid_image.add_(1).div_(2)

            # Save the grid image
            grid_image_path = all_dir / f'epoch_{self.current_epoch}_grid_image_{i}.jpg'
            save_image(grid_image, grid_image_path, nrow=1, normalize=False)

            

            if not skip_fid:
                real_imgs = nn.functional.interpolate(real_imgs, size=(299, 299), mode='bilinear', align_corners=False)
                real_imgs = real_imgs.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(real_imgs, real=True)

                gen_cf_c = nn.functional.interpolate(gen_cf_c, size=(299, 299), mode='bilinear', align_corners=False)
                gen_cf_c = gen_cf_c.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(gen_cf_c, real=False)

        num_samples = len(posterior_true)
        self.logger.info(f'Finished evaluating counterfactual results for epoch: {self.current_epoch}')

        cv_y_true, cv_y_pred = np.array(cv_y_true), np.array(cv_y_pred)
        cacc = np.mean(cv_y_true == cv_y_pred)
        self.logger.info(f'Counterfactual accuracy = {cacc} (num_samples={len(cv_y_pred)})')

        posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
        cv_score = np.mean(np.abs(posterior_true - posterior_pred) > tau)
        self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau}, num_samples={num_samples})')

        cf_iou_xc = self.val_iou_xc.compute().item()
        self.val_iou_xc.reset()
        self.logger.info(f'IoU(S, Sc) = {cf_iou_xc:.3f} (cf_thresh={self.cf_threshold}, num_samples={pred_num_abnormal_samples}, mask={self.cf_gt_seg_mask_idx})')

        cf_iou_xfx = self.val_iou_xfx.compute().item()
        self.val_iou_xfx.reset()
        self.logger.info(f'IoU(S, Sfx) = {cf_iou_xfx:.3f} (cf_thresh={self.cf_threshold}, num_samples={pred_num_abnormal_samples}, mask={self.cf_gt_seg_mask_idx})')

        fid_score = None
        if not skip_fid:
            fid_score = self.val_fid.compute().item()
            self.logger.info(f'FID(X, Xc) = {fid_score:.3f} (num_samples={num_samples}, features={self.fid_features})')
            self.val_fid.reset()

        self.logger.info(f'Ratio of true abnormal slices to classified as abnormal slices: {pred_num_abnormal_samples / (max(true_num_abnormal_samples, 1e-8))}')
        return {
            'counter_acc': cacc,
            f'cv_{int(tau*100)}': cv_score,
            'fid': fid_score,
            'cf_iou_xc': cf_iou_xc,
        }
