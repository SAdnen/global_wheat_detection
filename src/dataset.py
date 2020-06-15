import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


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

            if self.transforms:
                sample = self.transforms(**{"image": image,
                                            "bboxes": bboxes,
                                            "labels": labels})

                target["boxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32).reshape(-1, 4)
                image = sample["image"]
                return image, target
            else:
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
