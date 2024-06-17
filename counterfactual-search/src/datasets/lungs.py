from pathlib import Path

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchsampler import ImbalancedDatasetSampler


class LungsDataset(torch.utils.data.Dataset):
    

    def __init__(self, root_dir:str, split:str, model_n_classes, transforms:albu.Compose=None, explain_classifier: bool = True):
        print(model_n_classes)
        if model_n_classes==2:
            self.CLASSES = 'Normal', 'COVID'
            CLASSES = 'Normal', 'COVID'
        else:     

            # CLASSES = 'normal', 'lung_opacity', 'viral_pneumonia', 'covid'
            self.CLASSES = 'Lung_Opacity', 'Viral Pneumonia', 'Normal', 'COVID'
            CLASSES = 'Lung_Opacity', 'Viral Pneumonia', 'Normal', 'COVID'
        self.root_dir = Path(root_dir)
        self.split = split
        if explain_classifier:
            # dataset split for training/evaluation of counterfactual model
            ann_name = 'classifier_validation.csv' if split == 'train' else 'explanator_val.csv'
        else:
            # dataset split for training/evaluation of classification model
            ann_name = 'classifier_train.csv' if split == 'train' else 'classifier_validation.csv'
        ann_path = self.root_dir / ann_name
        print(CLASSES)
        print(self.CLASSES)
        self.labels, self.image_paths, self.mask_paths = self._load_anns(ann_path)
        self.labels = [self.CLASSES.index(lb) for lb in self.labels]
        self.transforms = transforms

    def _load_anns(self, ann_path):
        with open(ann_path) as fid:
            return zip(*(line.rstrip().split(',') for line in fid))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label, img_path, mask_path = self.labels[index], self.image_paths[index], self.mask_paths[index]
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        sample = {
            'image': img,
            'mask': mask,
        }
        if self.transforms:
            sample = self.transforms(**sample)
        sample['label'] = label
        sample['image'] = sample['image'][np.newaxis]
        sample['masks'] = (sample.pop('mask').clip(0, 1))[np.newaxis]
        return sample

    def get_sampling_labels(self):
        # required by https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
        return self.labels


def get_covid_dataloaders(data_transforms, params, sampler_labels=None):
    train_data = LungsDataset(Path(params.root_dir, params.annotations.train), data_transforms['train'])
    sampler_labels = train_data.get_sampling_labels() if sampler_labels is None else sampler_labels
    train_sampler = ImbalancedDatasetSampler(train_data, labels=sampler_labels) if params.get('use_sampler', True) else None

    test_data = LungsDataset(Path(params.root_dir, params.annotations.val), data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        batch_size=params.batch_size,
        pin_memory=True,
        num_workers=params.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=params.batch_size,
        shuffle=params.get('shuffle_test', False),
        pin_memory=True,
        num_workers=params.num_workers,
    )
    return train_loader, test_loader


if __name__ == '__main__':
    #DATA_PATH = Path('../COVID-19_Radiography_Dataset_v2/')
    DATA_PATH = Path("/gpfs/space/home/shoush/nn_project/project/datasets/COVID-19_Radiography_Dataset/")
    train_dst = LungsDataset(DATA_PATH / 'train.csv')
    sample = train_dst[0]
    plt.imshow(sample['image'][0], cmap='gray')
    plt.title(train_dst.CLASSES[sample['label']])
# from pathlib import Path

# import albumentations as albu
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from PIL import Image
# from torchsampler import ImbalancedDatasetSampler


# class LungsDataset(torch.utils.data.Dataset):
#     #CLASSES = 'normal', 'lung_opacity', 'viral_pneumonia', 'covid'
#     CLASSES = 'Lung_Opacity', 'Viral Pneumonia', 'Normal', 'COVID'

#     def __init__(self, root_dir:str, split:str, transforms:albu.Compose=None, explain_classifier: bool = True):
#         self.root_dir = Path(root_dir)
#         self.split = split
#         if explain_classifier:
#             # dataset split for training/evaluation of counterfactual model
#             ann_name = 'classifier_validation.csv' if split == 'train' else 'explanator_val.csv'
#         else:
#             # dataset split for training/evaluation of classification model
#             ann_name = 'classifier_train.csv' if split == 'train' else 'classifier_validation.csv'
#         ann_path = self.root_dir / ann_name
        

#         self.labels, self.image_paths, self.mask_paths = self._load_anns(ann_path)
#         self.labels = [self.CLASSES.index(lb) for lb in self.labels]
#         self.transforms = transforms
        
        
#         self.labels, self.image_paths, self.mask_paths = self._load_anns(ann_path)
#         self.labels = [self.CLASSES.index(lb) for lb in self.labels]
#         self.transforms = transforms
        
#         self.label_mapping = {
#             0: 1,  # 'Lung_Opacity' -> 'abnormal'
#             1: 1,  # 'Viral Pneumonia' -> 'abnormal'
#             2: 0,  # 'Normal' -> 'Normal'
#             3: 1   # 'COVID' -> 'abnormal'
#         }

#     def _load_anns(self, ann_path):
#         with open(ann_path) as fid:
#             return zip(*(line.rstrip().split(',') for line in fid))

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
        
#         original_label, img_path, mask_path = self.labels[index], self.image_paths[index], self.mask_paths[index]
#         label = self.label_mapping[original_label]
        
#         #label, img_path, mask_path = self.labels[index], self.image_paths[index], self.mask_paths[index]
#         # print(f"label: {label}, img_path: {img_path}, mask_path: {mask_path}")
#         img = np.array(Image.open(img_path))
#         mask = np.array(Image.open(mask_path))
#         if len(img.shape) == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         if len(mask.shape) == 3:
#             mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

#         sample = {
#             'image': img,
#             'mask': mask,
#         }
#         if self.transforms:
#             sample = self.transforms(**sample)
#         sample['label'] = label
#         sample['image'] = sample['image'][np.newaxis]
#         sample['masks'] = (sample.pop('mask').clip(0, 1))[np.newaxis]
#         return sample

#     def get_sampling_labels(self):
#         # required by https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
#         return self.labels


# def get_covid_dataloaders(data_transforms, params, sampler_labels=None):
#     train_data = LungsDataset(Path(params.root_dir, params.annotations.train), data_transforms['train'])
#     sampler_labels = train_data.get_sampling_labels() if sampler_labels is None else sampler_labels
#     train_sampler = ImbalancedDatasetSampler(train_data, labels=sampler_labels) if params.get('use_sampler', True) else None

#     test_data = LungsDataset(Path(params.root_dir, params.annotations.val), data_transforms['val'])
#     train_loader = torch.utils.data.DataLoader(
#         train_data,
#         sampler=train_sampler,
#         shuffle=train_sampler is None,
#         batch_size=params.batch_size,
#         pin_memory=True,
#         num_workers=params.num_workers,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=params.batch_size,
#         shuffle=params.get('shuffle_test', False),
#         pin_memory=True,
#         num_workers=params.num_workers,
#     )
#     return train_loader, test_loader


# if __name__ == '__main__':
#     # /home/mshoush/NN_Course_Raaul/project/datasets/COVID-19_Radiography_Dataset
#     DATA_PATH = Path("/gpfs/space/home/shoush/nn_project/project/datasets/COVID-19_Radiography_Dataset/")
#     # Path('../../../datasets/COVID-19_Radiography_Dataset/')
#     print("========================================\nHERE\n===================================")
#     print(f"\n+++++++++++++++++++++++++++\nDATA_PATH: {DATA_PATH}\n+++++++++++++++++++++++++++")
#     train_dst = LungsDataset(DATA_PATH / 'train.csv')
#     sample = train_dst[0]
#     plt.imshow(sample['image'][0], cmap='gray')
#     plt.title(train_dst.CLASSES[sample['label']])
