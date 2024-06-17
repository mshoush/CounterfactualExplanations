import cv2
import numpy as np
from skimage.measure import label   
import torch
from torch import nn
from tqdm import tqdm
from torchvision.utils import save_image
from src.models.cgan.counterfactual_inpainting_cgan import CounterfactualInpaintingCGAN
from src.visualizations import confmat_vis_img
from .counterfactual import CounterfactualTrainer
import os



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


def largest_cc(segmentation):
    labels = label(segmentation)
    if labels.max() == 0:
        return segmentation
    # assume at least 1 CC
    return (labels == np.argmax(np.bincount(labels.flat)[1:])+1).astype(np.uint8)


class CounterfactualInpaintingTrainer(CounterfactualTrainer):
    @torch.no_grad()
    def evaluate_counterfactual(self, loader, phase='val', tau=0.8, skip_fid=False, postprocess_morph=False):
        self.model.eval()

        cf_dir = self.cf_vis_dir_train if phase == 'train' else self.cf_vis_dir_val
        classes = []
        cv_y_true, cv_y_pred = [], []
        posterior_true, posterior_pred = [], []

        # Create directories for each type of image
        real_image_dir = cf_dir / 'real_images'
        generated_image_dir = cf_dir / 'generated_images'
        difference_image_dir = cf_dir / 'difference_images'
        all_dir = cf_dir / 'real_gen_diff_images'
        
        os.makedirs(real_image_dir, exist_ok=True)
        os.makedirs(generated_image_dir, exist_ok=True)
        os.makedirs(difference_image_dir, exist_ok=True)
        os.makedirs(all_dir, exist_ok=True)

        # number of samples where classifier predicted > 0.8 and the gt label is 1 (abnormal)
        pred_num_abnormal_samples = 0
        true_num_abnormal_samples = 0
        for i, batch in tqdm(enumerate(loader), desc='Validating counterfactuals', leave=False, total=len(loader)):
            # Evaluate Counterfactual Validity Metric
            real_imgs = batch['image'].cuda(non_blocking=True).float()
            real_image_path = real_image_dir / (f'epoch_{self.current_epoch}_real_image_atTheStart_{i}.png')
            save_image(real_imgs.data, real_image_path, nrow=1, normalize=False)
            
            cf_gt_masks = batch['masks'][:, self.cf_gt_seg_mask_idx].cuda(non_blocking=True).float()
            labels = batch['label']
            true_num_abnormal_samples += labels.sum()
            B = labels.shape[0]

            self.model: CounterfactualInpaintingCGAN
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.model.posterior_prob(real_imgs)

            # our ground truth is the `flipped` labels
            cv_y_true.extend(real_f_x_desired_discrete.cpu().squeeze(1).numpy())
            posterior_true.extend(real_f_x.cpu().squeeze(1).numpy())
            classes.extend(labels.cpu().numpy())

            # computes I_f(x, c)
            gen_cf_c = self.model.explanation_function(real_imgs, real_f_x_desired_discrete)

            # computes f(x_c)
            gen_f_x, gen_f_x_discrete, _, _ = self.model.posterior_prob(gen_cf_c)
            # our prediction is the classifier's label for the generated images given the desired posterior probability
            cv_y_pred.extend(gen_f_x_discrete.cpu().squeeze(1).numpy())
            posterior_pred.extend(gen_f_x.cpu().squeeze(1).numpy())

            # denorm values from [-1; 1] to [0, 1] range, B x 1 x H x W
            real_imgs.add_(1).div_(2)
            gen_cf_c.add_(1).div_(2)

            # compute difference maps, threshold and compute IoU
            # |x - x_c|
            diff = (real_imgs - gen_cf_c).abs() # [0; 1] values
            diff_seg = (diff > self.cf_threshold).byte()
            abnormal_mask = labels.bool()
            if abnormal_mask.any():
                if postprocess_morph:
                    diff_seg[abnormal_mask] = self.postprocess_morph(diff_seg[abnormal_mask])
                pred_num_abnormal_samples += abnormal_mask.sum()
                self.val_iou_xc.update(diff_seg[abnormal_mask].squeeze(1), cf_gt_masks[abnormal_mask])

            vis_confmat = confmat_vis_img(cf_gt_masks[0].unsqueeze(0).unsqueeze(0), diff_seg[0].unsqueeze(0), normalized=True)[0]
            vis = torch.stack((
                real_imgs[0], torch.zeros_like(real_imgs[0]), torch.zeros_like(real_imgs[0]), 
                gen_cf_c[0], diff[0], diff_seg[0],
            ), dim=0).permute(0, 2, 3, 1)
            vis = torch.cat((vis, vis, vis), 3)
            vis[1] = 0.3 * vis[0] + 0.7 * vis_confmat
            vis[2] = vis_confmat
            vis = vis.permute(0, 3, 1, 2)

            # save first example for visualization
            vis_path = cf_dir / (f'epoch_%d_counterfactual_%d_label_%d_true_%d_pred_%d.png' % (
                self.current_epoch, i, labels[0], real_f_x_desired_discrete[0][0], gen_f_x_discrete[0][0])
            )
            save_image(vis.data, vis_path, nrow=3, normalize=False) # value_range=(-1, 1))

            # Save the real image, generated image, and difference image in separate directories
            real_image_path = real_image_dir / (f'epoch_{self.current_epoch}_real_image_{i}.png')
            generated_image_path = generated_image_dir / (f'epoch_{self.current_epoch}_generated_image_{i}.png')
            difference_image_path = difference_image_dir / (f'epoch_{self.current_epoch}_difference_image_{i}.png')

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
                # Evaluate Frechet Inception Distance (FID)
                # upsample to InceptionV3's resolution and convert to RGB
                real_imgs = nn.functional.interpolate(real_imgs, size=(299, 299), mode='bilinear', align_corners=False)
                real_imgs = real_imgs.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(real_imgs, real=True)

                # upsample to InceptionV3's resolution and convert to RGB
                gen_cf_c = nn.functional.interpolate(gen_cf_c, size=(299, 299), mode='bilinear', align_corners=False)
                gen_cf_c = gen_cf_c.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(gen_cf_c, real=False)

        num_samples = len(posterior_true)
        self.logger.info(f'Finished evaluating counterfactual results for epoch: {self.current_epoch}')

        # Counterfactual Accuracy (flip rate) Score
        cv_y_true, cv_y_pred = np.array(cv_y_true), np.array(cv_y_pred)
        cacc = np.mean(cv_y_true == cv_y_pred)
        self.logger.info(f'Counterfactual accuracy = {cacc} (num_samples={len(cv_y_pred)})')

        posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
        # considering a flip rate from positives to negatives only where f(x) > tau
        pos_true_mask = posterior_true > tau
        # Counterfactual Validity Score
        cv_score = np.mean(np.abs(posterior_true - posterior_pred)[pos_true_mask] > tau)
        self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau}, num_samples={pos_true_mask.sum()})')

        cf_iou_xc = self.val_iou_xc.compute().item()
        self.val_iou_xc.reset()
        self.logger.info(f'IoU(S, Sc) = {cf_iou_xc:.3f} (cf_thresh={self.cf_threshold}, num_samples={pred_num_abnormal_samples}, mask={self.cf_gt_seg_mask_idx})')

        fid_score = None
        if not skip_fid:
            # Frechet Inception Distance (FID) Score
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


    
#     def evaluate_counterfactual(self, loader, phase='val', tau=0.8, skip_fid=False, postprocess_morph:bool=False):
#         self.model.eval()
        
#         cf_dir = self.cf_vis_dir_train if phase == 'train' else self.cf_vis_dir_val
#         classes = []
#         cv_y_true, cv_y_pred = [], []
#         posterior_true, posterior_pred = [], []

#         # number of samples where classifier predicted > 0.8 and the gt label is 1 (abnormal)
#         pred_num_abnormal_samples = 0
#         true_num_abnormal_samples = 0
#         for i, batch in tqdm(enumerate(loader), desc='Validating counterfactuals', leave=False, total=len(loader)):
#             # Evaluate Counterfactual Validity Metric
#             real_imgs = batch['image'].cuda(non_blocking=True).float()
#             cf_gt_masks = batch['masks'][:, self.cf_gt_seg_mask_idx].cuda(non_blocking=True).float()
#             labels = batch['label']
#             true_num_abnormal_samples += labels.sum()
#             B = labels.shape[0]

#             self.model: CounterfactualInpaintingCGAN
#             real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.model.posterior_prob(real_imgs)

#             # our ground truth is the `flipped` labels
#             cv_y_true.extend(real_f_x_desired_discrete.cpu().squeeze(1).numpy())
#             posterior_true.extend(real_f_x.cpu().squeeze(1).numpy())
#             classes.extend(labels.cpu().numpy())

#             # computes I_f(x, c)
#             gen_cf_c = self.model.explanation_function(real_imgs, real_f_x_desired_discrete)

#             # computes f(x_c)
#             gen_f_x, gen_f_x_discrete, _, _ = self.model.posterior_prob(gen_cf_c)
#             # our prediction is the classifier's label for the generated images given the desired posterior probability
#             cv_y_pred.extend(gen_f_x_discrete.cpu().squeeze(1).numpy())
#             posterior_pred.extend(gen_f_x.cpu().squeeze(1).numpy())

#             # denorm values from [-1; 1] to [0, 1] range, B x 1 x H x W
#             real_imgs.add_(1).div_(2)
#             gen_cf_c.add_(1).div_(2)

#             # compute difference maps, threshold and compute IoU
#             # |x - x_c|
#             diff = (real_imgs - gen_cf_c).abs() # [0; 1] values
#             diff_seg = (diff > self.cf_threshold).byte()
#             abnormal_mask = labels.bool()
#             if abnormal_mask.any():
#                 if postprocess_morph:
#                     diff_seg[abnormal_mask] = self.postprocess_morph(diff_seg[abnormal_mask])
#                 pred_num_abnormal_samples += abnormal_mask.sum()
#                 self.val_iou_xc.update(diff_seg[abnormal_mask].squeeze(1), cf_gt_masks[abnormal_mask])

#             vis_confmat = confmat_vis_img(cf_gt_masks[0].unsqueeze(0).unsqueeze(0), diff_seg[0].unsqueeze(0), normalized=True)[0]
#             vis = torch.stack((
#                 real_imgs[0], torch.zeros_like(real_imgs[0]), torch.zeros_like(real_imgs[0]), 
#                 gen_cf_c[0], diff[0], diff_seg[0],
#             ), dim=0).permute(0, 2, 3, 1)
#             vis = torch.cat((vis, vis, vis), 3)
#             vis[1] = 0.3*vis[0] + 0.7 * vis_confmat
#             vis[2] = vis_confmat
#             vis = vis.permute(0, 3, 1, 2)
            
#             # save first example for visualization
#             vis_path = cf_dir / (f'epoch_%d_counterfactual_%d_label_%d_true_%d_pred_%d.png' % (
#                 self.current_epoch, i, labels[0], real_f_x_desired_discrete[0][0], gen_f_x_discrete[0][0])
#             )
#             save_image(vis.data, vis_path, nrow=3, normalize=False) # value_range=(-1, 1))

#             if not skip_fid:
#                 # Evaluate Frechet Inception Distance (FID)
#                 # upsample to InceptionV3's resolution and convert to RGB
#                 real_imgs = nn.functional.interpolate(real_imgs, size=(299, 299), mode='bilinear', align_corners=False)
#                 real_imgs = real_imgs.repeat_interleave(repeats=3, dim=1)
#                 self.val_fid.update(real_imgs, real=True)
                
#                 # upsample to InceptionV3's resolution and convert to RGB
#                 gen_cf_c = nn.functional.interpolate(gen_cf_c, size=(299, 299), mode='bilinear', align_corners=False)
#                 gen_cf_c = gen_cf_c.repeat_interleave(repeats=3, dim=1)
#                 self.val_fid.update(gen_cf_c, real=False)

#         num_samples = len(posterior_true)
#         self.logger.info(f'Finished evaluating counterfactual results for epoch: {self.current_epoch}')

#         # Counterfactual Accuracy (flip rate) Score
#         cv_y_true, cv_y_pred = np.array(cv_y_true), np.array(cv_y_pred)
#         cacc = np.mean(cv_y_true == cv_y_pred)
#         self.logger.info(f'Counterfactual accuracy = {cacc} (num_samples={len(cv_y_pred)})')

#         posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
#         # considering a flip rate from positives to negatives only where f(x) > tau
#         pos_true_mask = posterior_true > tau
#         # Counterfactual Validity Score
#         cv_score = np.mean(np.abs(posterior_true - posterior_pred)[pos_true_mask] > tau)
#         self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau}, num_samples={pos_true_mask.sum()})')

#         cf_iou_xc = self.val_iou_xc.compute().item()
#         self.val_iou_xc.reset()
#         self.logger.info(f'IoU(S, Sc) = {cf_iou_xc:.3f} (cf_thresh={self.cf_threshold}, num_samples={pred_num_abnormal_samples}, mask={self.cf_gt_seg_mask_idx})')

#         fid_score = None
#         if not skip_fid:
#             # Frechet Inception Distance (FID) Score
#             fid_score = self.val_fid.compute().item()
#             self.logger.info(f'FID(X, Xc) = {fid_score:.3f} (num_samples={num_samples}, features={self.fid_features})')
#             self.val_fid.reset()
        
#         self.logger.info(f'Ratio of true abnormal slices to classified as abnormal slices: {pred_num_abnormal_samples / (max(true_num_abnormal_samples, 1e-8))}')
#         return {
#             'counter_acc': cacc,
#             f'cv_{int(tau*100)}': cv_score,
#             'fid': fid_score,
#             'cf_iou_xc': cf_iou_xc,
#         }
    
    def postprocess_morph(self, masks:torch.Tensor):
        masks_np = masks.cpu().numpy().squeeze(1)
        kernel = np.ones((3, 3),np.uint8)

        for i, mask in enumerate(masks_np):
            # remove small objects
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # remove small holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # select only largest connected commonent
            masks_np[i] = largest_cc(mask)
        return torch.from_numpy(masks_np).type_as(masks).unsqueeze(1)


class CounterfactualInpaintingV2Trainer(CounterfactualInpaintingTrainer):
    @torch.no_grad()
    def evaluate_counterfactual(self, loader, phase='val', tau=0.8, skip_fid=False, postprocess_morph:bool=False):
        self.model.eval()
        
        cf_dir = self.cf_vis_dir_train if phase == 'train' else self.cf_vis_dir_val
        classes = []
        cv_y_true, cv_y_pred = [], []
        posterior_true, posterior_pred = [], []

        # number of samples where classifier predicted > 0.8 and the gt label is 1 (abnormal)
        pred_num_abnormal_samples = 0
        true_num_abnormal_samples = 0
        print(len(loader))
        for i, batch in tqdm(enumerate(loader), desc='Validating counterfactuals', leave=False, total=len(loader)):
            # Evaluate Counterfactual Validity Metric
            real_imgs = batch['image'].cuda(non_blocking=True).float()
            cf_gt_masks = batch['masks'][:, self.cf_gt_seg_mask_idx].cuda(non_blocking=True).float()
            labels = batch['label']
            true_num_abnormal_samples += labels.sum()
            B = labels.shape[0]

            self.model: CounterfactualInpaintingCGAN
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.model.posterior_prob(real_imgs)

            # our ground truth is the `flipped` labels
            cv_y_true.extend(real_f_x_desired_discrete.cpu().squeeze(1).numpy())
            posterior_true.extend(real_f_x.cpu().squeeze(1).numpy())
            classes.extend(labels.cpu().numpy())

            # computes I_f(x, c)
            gen_cf_c = self.model.explanation_function(real_imgs, real_f_x_discrete)

            # computes f(x_c)
            gen_f_x, gen_f_x_discrete, _, _ = self.model.posterior_prob(gen_cf_c)
            # our prediction is the classifier's label for the generated images given the desired posterior probability
            cv_y_pred.extend(gen_f_x_discrete.cpu().squeeze(1).numpy())
            posterior_pred.extend(gen_f_x.cpu().squeeze(1).numpy())

            # denorm values from [-1; 1] to [0, 1] range, B x 1 x H x W
            real_imgs.add_(1).div_(2)
            gen_cf_c.add_(1).div_(2)

            # compute difference maps, threshold and compute IoU
            # |x - x_c|
            diff = (real_imgs - gen_cf_c).abs() # [0; 1] values
            diff_seg = (diff > self.cf_threshold).byte()
            abnormal_mask = labels.bool()
            if abnormal_mask.any():
                if postprocess_morph:
                    diff_seg[abnormal_mask] = self.postprocess_morph(diff_seg[abnormal_mask])
                pred_num_abnormal_samples += abnormal_mask.sum()
                self.val_iou_xc.update(diff_seg[abnormal_mask].squeeze(1), cf_gt_masks[abnormal_mask])

            vis_confmat = confmat_vis_img(cf_gt_masks[0].unsqueeze(0).unsqueeze(0), diff_seg[0].unsqueeze(0), normalized=True)[0]
            vis = torch.stack((
                real_imgs[0], torch.zeros_like(real_imgs[0]), torch.zeros_like(real_imgs[0]), 
                gen_cf_c[0], diff[0], diff_seg[0],
            ), dim=0).permute(0, 2, 3, 1)
            vis = torch.cat((vis, vis, vis), 3)
            vis[1] = 0.3*vis[0] + 0.7 * vis_confmat
            vis[2] = vis_confmat
            vis = vis.permute(0, 3, 1, 2)
            
            # save first example for visualization
            vis_path = cf_dir / (f'epoch_%d_counterfactual_%d_label_%d_true_%d_pred_%d.png' % (
                self.current_epoch, i, labels[0], real_f_x_desired_discrete[0][0], gen_f_x_discrete[0][0])
            )
            save_image(vis.data, vis_path, nrow=3, normalize=False) # value_range=(-1, 1))

            if not skip_fid:
                # Evaluate Frechet Inception Distance (FID)
                # upsample to InceptionV3's resolution and convert to RGB
                real_imgs = nn.functional.interpolate(real_imgs, size=(299, 299), mode='bilinear', align_corners=False)
                real_imgs = real_imgs.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(real_imgs, real=True)
                
                # upsample to InceptionV3's resolution and convert to RGB
                gen_cf_c = nn.functional.interpolate(gen_cf_c, size=(299, 299), mode='bilinear', align_corners=False)
                gen_cf_c = gen_cf_c.repeat_interleave(repeats=3, dim=1)
                self.val_fid.update(gen_cf_c, real=False)

        num_samples = len(posterior_true)
        self.logger.info(f'Finished evaluating counterfactual results for epoch: {self.current_epoch}')

        # Counterfactual Accuracy (flip rate) Score
        cv_y_true, cv_y_pred = np.array(cv_y_true), np.array(cv_y_pred)
        cacc = np.mean(cv_y_true == cv_y_pred)
        self.logger.info(f'Counterfactual accuracy = {cacc} (num_samples={len(cv_y_pred)})')

        posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
        # considering a flip rate from positives to negatives only where f(x) > tau
        pos_true_mask = posterior_true > tau
        # Counterfactual Validity Score
        cv_score = np.mean(np.abs(posterior_true - posterior_pred)[pos_true_mask] > tau)
        self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau}, num_samples={pos_true_mask.sum()})')

        cf_iou_xc = self.val_iou_xc.compute().item()
        self.val_iou_xc.reset()
        self.logger.info(f'IoU(S, Sc) = {cf_iou_xc:.3f} (cf_thresh={self.cf_threshold}, num_samples={pred_num_abnormal_samples}, mask={self.cf_gt_seg_mask_idx})')

        fid_score = None
        if not skip_fid:
            # Frechet Inception Distance (FID) Score
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
