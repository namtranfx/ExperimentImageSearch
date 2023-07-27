import os
import json
import random


from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms

from PIL import Image

import pandas as pd
from imutils import paths
import tqdm

# from source.data_handler import MyTransform

#from source.data_handler import MyTransform


class TripletData(Dataset):
    def __init__(self, path, transforms, split="train"):
        self.path = path
        self.split = split    # train or valid
        self.cats = 102       # number of categories
        self.transforms = transforms
    
    def __getitem__(self, idx):
        # our positive class for the triplet
        idx = str(idx%self.cats + 1)
        
        # choosing our pair of positive images (im1, im2)
        positives = os.listdir(os.path.join(self.path, idx))
        im1, im2 = random.sample(positives, 2)
        
        # choosing a negative class and negative image (im3)
        negative_cats = [str(x+1) for x in range(self.cats)]
        negative_cats.remove(idx)
        negative_cat = str(random.choice(negative_cats))
        negatives = os.listdir(os.path.join(self.path, negative_cat))
        im3 = random.choice(negatives)
        
        im1,im2,im3 = os.path.join(self.path, idx, im1), os.path.join(self.path, idx, im2), os.path.join(self.path, negative_cat, im3)
        
        im1 = self.transforms(Image.open(im1))
        im2 = self.transforms(Image.open(im2))
        im3 = self.transforms(Image.open(im3))
        
        return [im1, im2, im3]
        
    # we'll put some value that we want since there can be far too many triplets possible
    # multiples of the number of images/ number of categories is a good choice
    def __len__(self):
        return self.cats*8
    
class MyTransform_norm:
    def __init__(self):

        
        # Transforms
        self.train_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    def process(self, img):
        return self.train_transforms(img)


class MyTransform:
    def __init__(self) -> None:
        self.transforms_no_norm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    def process(self, img: Image):
        return self.transforms_no_norm(img=img)

################################################################################
#############################  CUSTOM DATASET CLASS ############################
################################################################################

#------------------------------------------------------
from pycocotools.coco import COCO


class CustomCocoDataset(Dataset):
    def __init__(self, root, annFile, transform : MyTransform):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        cat_ids = [ann['category_id'] for ann in anns]
        cats = coco.loadCats(cat_ids)
        cat_names = [cat['name'] for cat in cats]

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = f'{self.root}/{path}'
        img = Image.open(img_path).convert('RGB')
        img = self.transform.process(img)

        return img, cat_names, img_path

    def __len__(self):
        return len(self.ids)

#-----------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, transform:MyTransform) -> None:
        self.imgpath = []
        self.labels = []
        self.transform = transform
    def __getitem__(self, index) -> tuple:
        return self.transform.process(Image.open(self.imgpath[index]).convert('RGB')), self.labels[index], self.imgpath[index]
    def __len__(self) -> int:
        return len(self.imgpath)
    def getLabels(self):
        return self.labels

class Caltech101_temp(CustomDataset):
    def __init__(self, datasetpath, transform: MyTransform) -> None:
        super().__init__(transform)
        image_paths = list(paths.list_images(datasetpath))
        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]
            if label == "BACKGROUND_Google":
                continue
            self.imgpath.append(img_path)
            self.labels.append(label)
class CaltechDataset(CustomDataset):
    def __init__(self, datasetpath, transform) -> None:
        super().__init__(transform)
        
        for obj_name in os.listdir(datasetpath):
            curr_path = os.path.join(datasetpath, obj_name)
            if os.path.isdir(curr_path):
                for filename in os.listdir(curr_path):
                    self.imgpath.append(os.path.join(curr_path, filename))
                    self.labels.append(obj_name)

class Oxford102Flower(CustomDataset):
    def __init__(self, datasetpath, transform: MyTransform) -> None:
        super().__init__(transform)

        for obj_name in os.listdir(datasetpath):
            curr_path = os.path.join(datasetpath, obj_name)
            if os.path.isdir(curr_path):
                for filename in os.listdir(curr_path):
                    self.imgpath.append(os.path.join(curr_path, filename))
                    self.labels.append(obj_name)


class CorelDataset(CustomDataset):
    def __init__(self, datasetpath, transform) -> None:
        super().__init__(transform)

        for filename in os.listdir(datasetpath):
            id = int(filename.split(".")[0].split("_")[0])
            label = int(id/100)
            self.imgpath.append(os.path.join(datasetpath, filename))
            self.labels.append(label)
class OxfordDataset(CustomDataset):
    def __init__(self, datasetpath, transform) -> None:
        super().__init__(transform)
        
        for filename in os.listdir(datasetpath):
            label = filename.split("_0")[0]
            self.imgpath.append(os.path.join(datasetpath, filename))
            self.labels.append(label)
    

class InriaHolidayDataset(CustomDataset):
    def __init__(self, datasetpath, transform) -> None:
        super().__init__(transform)

        for filename in os.listdir(datasetpath):
            id = int(filename.split(".")[0])
            label = int(id/100)
            self.imgpath.append(os.path.join(datasetpath, filename))
            self.labels.append(label)            

class CifarDataset(CustomDataset):
    def __init__(self, datasetpath, transform: MyTransform) -> None:
        super().__init__(transform)

        csv_label = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\cifar-10\\trainLabels.csv"
        label_dataframe = pd.read_csv(csv_label)

        for idx_row in range(0, label_dataframe.shape[0], 1):
            id = label_dataframe.iloc[idx_row][label_dataframe.columns.values[0]]
            label = label_dataframe.iloc[idx_row][label_dataframe.columns.values[1]]
            
            self.imgpath.append(os.path.join(datasetpath, str(id) + ".png"))
            self.labels.append(label)
