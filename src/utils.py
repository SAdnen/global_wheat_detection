import albumentations as A
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from src.dataset import GlobalWheatDataset, CutMixDataset
from src.config import path_settings
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch

def merge_targets(target1, target2):
    targets = []
    for i in range(len(target1)):
        target = dict()
        for key in target1[i].keys():
            merged_value = torch.cat([target1[i][key], target2[i][key]])
            target[key] = merged_value
        targets.append(target)
    return tuple(targets)

def mixup_images(images1, images2):
    mixed_images = [0.5 * (images1 + images2) for (images1, images2) in zip(images1, images2)]
    return tuple(mixed_images)

def random_crop(image, target):
    boxes_crop = target["boxes"]
    x, y, _ = image.size()
    xc, yc = x//2, y//2
    image_crop = image[: xc, : yc, :]
    boxes_crop[:, [0, 2]] = np.clip(boxes_crop[:, [0, 2]], 0, xc)
    boxes_crop[:, [1, 3]] = np.clip(boxes_crop[:, [1, 3]], 0, yc)
    mask = (boxes_crop[:, 0]<xc) * (boxes_crop[:, 1]<yc)
    boxes_crop = boxes_crop[mask]

    area = (boxes_crop[:, 2] - boxes_crop[:, 0]) * (boxes_crop[:, 3] - boxes_crop[:, 1])
    labels = torch.ones(len(boxes_crop), dtype=torch.int64)
    iscrowd = torch.zeros(len(boxes_crop), dtype=torch.uint8)
    target["boxes"] = boxes_crop
    target["area"] = area
    target["iscrowd"] = iscrowd
    target["labels"] = labels

    return image_crop, target


def cutmix_images(image, target, image_crop, target_crop):
    _, x, y = image.size()
    _, xc, yc = image_crop.size()

    # xp = np.random.randint(0, x-xc)
    # yp = np.random.randint(0, y-yc)
    xp = 0
    yp = 0
    image[xp:xp+xc, yp:yp+yc, :] = image_crop
    target_crop["boxes"] = adjust_boxes(target_crop, xp, yp)

    boxes = target["boxes"].copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], xp, xp+xc)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], yp, yp+yc)
    mask = (xp<boxes[:, 0]<xc+xp) * (xp<boxes[:, 1]<yc+yp)
    target["boxes"] = target["boxes"][np.logical_not(mask)]
    target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (target["boxes"][:, 3] - target["boxes"][:, 1])
    labels = torch.ones(len(target["boxes"]), dtype=torch.int64)
    iscrowd = torch.zeros(len(labels), dtype=torch.uint8)
    target["iscrowd"] = iscrowd
    target["labels"] = labels

    return image, target, target_crop


def adjust_boxes(target, xp, yp):
    boxes = target["boxes"]
    boxes[:, [0, 2]] += xp
    boxes[:, [1, 3]] += yp
    return boxes


def get_train_transforms(cfg):
    image_size = cfg["image_size"]
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(image_size-200, image_size-200), height=image_size, width=image_size, p=0.5),

            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
                A.GaussNoise(var_limit=(0.01, .005), mean=0, always_apply=False, p=0.6),
            ], p=0.9),

            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=image_size, width=image_size, p=1),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=128, fill_value=0, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=128, max_width=32, fill_value=0, p=0.3),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms(cfg):
    image_size = cfg["image_size"]
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_test_transforms(cfg):
    image_size = cfg["image_size"]
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def collate_fn(batch):
    sample = tuple(zip(*batch))
    return sample


def get_train_val_indexes(folds_df, ifold):
    val_df = folds_df.loc[folds_df.fold == ifold]
    train_df = folds_df.loc[folds_df.fold != ifold]
    return train_df.image_id.values, val_df.image_id.values


def get_train_valid_dataloaders(ifold: int, df: pd.DataFrame, folds_df: pd.DataFrame, train_dir: str, cfg: dict) -> [DataLoader, DataLoader]:
    train_transforms = get_train_transforms(cfg)
    valid_transforms = get_valid_transforms(cfg)
    train_idx, valid_idx = get_train_val_indexes(folds_df, ifold)
    train_dataset = GlobalWheatDataset(df, train_idx, train_dir, train_transforms, train=True)
    valid_dataset = GlobalWheatDataset(df, valid_idx, train_dir, valid_transforms, train=True)
    cutmix_dataset = CutMixDataset(df, train_idx, train_dir, train_transforms, train=True)
    cutmix_dataloader = DataLoader(cutmix_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True,
                                  collate_fn=collate_fn)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True,
                                  collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    train_dataloader_mix = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True,
                                  collate_fn=collate_fn)
    if cfg['mixup'] == True:
        return zip(train_dataloader, train_dataloader_mix), valid_dataloader
    elif cfg['cutmix'] == True:
        return cutmix_dataloader, valid_dataloader
    else:
        return train_dataloader, valid_dataloader


def get_test_dataloader(test_df, test_dir, batch_size, cfg):
    test_transforms = get_test_transforms(cfg)
    test_dataset = GlobalWheatDataset(test_df, test_df.image_id.unique(), test_dir, test_transforms, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
    return test_dataloader


def skFold(df: pd.DataFrame, nfolds: int=5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=7)
    folds_df = df.groupby(["image_id", "source"])["source"].count().to_frame(name="bbox_count").reset_index()
    folds_df['stratify_group'] = np.char.add(folds_df["source"].values.astype(str), folds_df['bbox_count'].apply(lambda x: f'_{x // 20}').values.astype(str))
    folds_df['fold'] = 0
    for fold, (train_index, test_index) in enumerate(skf.split(folds_df, folds_df.stratify_group)):
        folds_df.loc[test_index, 'fold'] = fold
    return folds_df


def get_bboxes_areas(row_box):
    bbox = np.fromstring(row_box[1:-1], sep=",")
    x, y, w, h = bbox
    return x, y, x+w, y+h, w, h, w*h

def load_dataframes():
    df = pd.read_csv("../train.csv")
    df['xmin'] = -1
    df['ymin'] = -1
    df['xmax'] = -1
    df['ymax'] = -1
    df['w'] = -1
    df['h'] = -1
    df['area'] = 0
    df[['xmin', 'ymin', 'xmax', 'ymax', 'w', 'h', 'area']] = np.stack(
        df['bbox'].apply(lambda row_box: get_bboxes_areas(row_box)))
    df.drop(columns=["bbox"], inplace=True)
    df = df.drop(df.loc[(df.w < 20) | (df.h < 20)].index.values, axis=0)
    test_df = pd.read_csv("../sample_submission.csv")
    return df, test_df


class Dloaders:
    df, test_df = load_dataframes()
    #folds_df = skFold(df)
    def get_dataloaders(self, ifold, cfg):
        folds_df = pd.read_csv(cfg["folds_df"])
        train_dir = path_settings["train"]
        test_dir = path_settings["test"]
        train_dataloader, valid_dataloader = get_train_valid_dataloaders(ifold, self.df, folds_df, train_dir, cfg)
        test_datalaoder = get_test_dataloader(self.test_df, test_dir, batch_size=2, cfg=cfg)
        data_loaders = {"train": train_dataloader,
                        "valid": valid_dataloader,
                        "test": test_datalaoder}
        return data_loaders




if __name__ == "__main__":
    images = torch.rand((5, 3, 10, 10))
    image_crops = torch.zeros((5, 3, 5, 5))
    cutmix_images = cutmix_images(images, image_crops)
    print("done")
