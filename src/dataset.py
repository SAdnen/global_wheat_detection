import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset

def merge_targets(target1, target2):
    target = dict()
    for key in target1.keys():
        if key == "boxes":
            merged_value = np.concatenate([target1[key], target2[key]])
        elif key == "image_id":
            merged_value = target1[key]
        else:
            merged_value = torch.cat([target1[key], target2[key]])
        #print(key, len(merged_value))
        target[key] = merged_value
    return target

def mixup_images(images1, images2):
    mixed_images = [0.5 * (images1 + images2) for (images1, images2) in zip(images1, images2)]
    return tuple(mixed_images)

def random_crop(image, target):
    boxes_crop = target["boxes"]
    x, y, _ = image.shape
    xc, yc = x//2, y//2
    image_crop = image[: xc, : yc, :]
    boxes_crop[:, [0, 2]] = np.clip(boxes_crop[:, [0, 2]], 0, xc)
    boxes_crop[:, [1, 3]] = np.clip(boxes_crop[:, [1, 3]], 0, yc)
    mask = (boxes_crop[:, 0]<xc-5) * (boxes_crop[:, 1]<yc-5)
    boxes_crop = boxes_crop[mask]

    area = (boxes_crop[:, 2] - boxes_crop[:, 0]) * (boxes_crop[:, 3] - boxes_crop[:, 1])
    labels = torch.ones(len(boxes_crop), dtype=torch.int64)
    iscrowd = torch.zeros(len(boxes_crop), dtype=torch.uint8)
    target["boxes"] = boxes_crop
    target["area"] = torch.as_tensor(area, dtype=torch.float32)
    target["iscrowd"] = iscrowd
    target["labels"] = labels

    return image_crop, target


def cutmix_images(image, target, image_crop, target_crop):
    x, y, _ = image.shape
    xc, yc, _ = image_crop.shape

    xp = 0 #np.random.randint(0, x-xc)
    yp = 0#np.random.randint(0, y-yc)

    image[xp:xp+xc, yp:yp+yc, :] = image_crop
    target_crop["boxes"] = adjust_boxes(target_crop, xp, yp)

    boxes = target["boxes"].copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], xp, xp+xc)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], yp, yp+yc)
    mask = (xp<boxes[:, 0]) * (boxes[:, 0]<xc+xp) * (yp<boxes[:, 1]) * (boxes[:, 1]<yc+yp)
    target["boxes"] = target["boxes"][np.logical_not(mask)]
    area = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (target["boxes"][:, 3] - target["boxes"][:, 1])
    labels = torch.ones(len(target["boxes"]), dtype=torch.int64)
    iscrowd = torch.zeros(len(labels), dtype=torch.uint8)
    target["iscrowd"] = iscrowd
    target["labels"] = labels
    target["area"] = torch.as_tensor(area, dtype=torch.float32)

    return image, target, target_crop


def adjust_boxes(target, xp, yp):
    boxes = target["boxes"]
    boxes[:, [0, 2]] += xp
    boxes[:, [1, 3]] += yp
    return boxes



class GlobalWheatDataset(Dataset):
    def __init__(self, df, image_ids, data_dir, transforms, train=True):
        self.df = df
        self.image_ids = image_ids
        self.data_dir = data_dir
        self.transforms = transforms
        self.train = train

    def __len__(self,):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        filename = image_id + ".jpg"
        image_path = os.path.join(self.data_dir, filename)
        image = self.load_image(image_path)
        if self.train:
            bboxes_areas = self.df[self.df.image_id==image_id][["xmin", "ymin", "xmax", "ymax", "area"]].values  # .astype(np.float)
            # bboxes_areas = torch.as_tensor(bboxes_areas, dtype=torch.float32)
            bboxes = bboxes_areas[:, :-1]
            area = torch.as_tensor(bboxes_areas[:, -1])
            labels = torch.ones(len(bboxes), dtype=torch.int64)
            image_id = torch.as_tensor([index], dtype=torch.int64)
            iscrowd = torch.zeros(len(bboxes_areas), dtype=torch.uint8)

            target = {"boxes": bboxes,
                      "labels": labels,
                      "image_id": image_id,
                      "area": area,
                      "iscrowd": iscrowd,
                      }

            sample = self.transforms(**{"image": image,
                                        "bboxes": bboxes,
                                        "labels": labels})

            target["boxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
            image = sample["image"]
            return image, target
        else:
            sample = self.transforms(**{"image": image})
            return sample["image"], image_id


    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        #         image = Image.open(image_path).convert("RGB")
        return image

class TestDataset(Dataset):
    def __init__(self, df, root_dir, transforms):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df.loc[index, 'image_id']
        image_path = os.path.join(self.root_dir, image_id +".jpg")
        image = self.load(image_path)
        sample = self.transforms(**{"image": image})
        return sample["image"], image_id

    def load(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image


class CutMixDataset(GlobalWheatDataset):
    def __init__(self, df, image_ids, data_dir, transforms, train=True):
        super(CutMixDataset, self).__init__(df, image_ids, data_dir, transforms, train=True)

    def __getitem__(self, index):
        rindex = np.random.randint(0, len(self.image_ids))
        image, target = self.getitem(index)
        rimage, rtarget = self.getitem(rindex)
        rimage_crop, rtarget = random_crop(rimage, rtarget)
        image, target, target_crop = cutmix_images(image, target, rimage_crop, rtarget)
        target = merge_targets(target, target_crop)

        if self.train:
            sample = self.transforms(**{"image": image,
                                        "bboxes": target["boxes"],
                                        "labels": target["labels"]})

            target["boxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
            image = sample["image"]

            return image, target

    def getitem(self, index):
        image_id = self.image_ids[index]
        filename = image_id + ".jpg"
        image_path = os.path.join(self.data_dir, filename)
        image = self.load_image(image_path)

        bboxes_areas = self.df[self.df.image_id == image_id][
            ["xmin", "ymin", "xmax", "ymax", "area"]].values  # .astype(np.float)
        bboxes = bboxes_areas[:, :-1]

        area = torch.as_tensor(bboxes_areas[:, -1])
        labels = torch.ones(len(bboxes), dtype=torch.int64)
        image_id = torch.as_tensor([index], dtype=torch.int64)
        iscrowd = torch.zeros(len(bboxes_areas), dtype=torch.uint8)

        target = {"boxes": bboxes,
                  "labels": labels,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd,
                  }
        return image, target

if __name__ == "__main__":
    print("done")