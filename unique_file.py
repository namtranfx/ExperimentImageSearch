import os
import random


from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms




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
    
class MyTransforms:
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
##############################################################################################
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    # Distances in embedding space is calculated in euclidean
    def forward(self, anchor, positive, negative):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
##############################################################################################
from torchvision import models
import torch.optim as optim
from tqdm import tqdm



# Our base model
class ResDeepFeature:
    def __init__(self, device):
        self._device = device
        self.model = models.resnet18().cpu()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.triplet_loss = TripletLoss()
    
    def trainDescriptor(self, train_loader):
        print("Training our Desciptor")
        epochs = 2
        # Training
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            print("Epoch ", epoch)
            for data in tqdm(train_loader):
                print("cannot go in for loop")
                self.optimizer.zero_grad()
                x1,x2,x3 = data
                e1 = self.model(x1.cpu())
                e2 = self.model(x2.cpu())
                e3 = self.model(x3.cpu()) 
                
                loss = self.triplet_loss(e1,e2,e3)
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
            print("Train Loss: {}".format(epoch_loss.item()))
        print("Training completed!")

    def extractFeature(self, img):
        return self.model(img)
    

##############################################################################################
import os

import glob
import faiss 
from PIL import Image
import numpy as np



class FlowerImageSearch:
    def __init__(self) -> None:
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._m_data_transform = MyTransforms()
        self._m_model = ResDeepFeature(self._device)
        self._faiss_index = faiss.IndexFlatL2(1000)   # build the index
        self._im_indices = []

    def trainDescriptor(self, data_loader):
        self._m_model.trainDescriptor(data_loader)
        


    def indexing(self, db_path):
        print("Indexing your database...")
        with torch.no_grad():
            for f in glob.glob(os.path.join(db_path, '*/*')):
                im = Image.open(f)
                im = im.resize((224,224))
                im = torch.tensor([self._m_data_transform.val_transforms(im).numpy()]).to(device=self._device)
            
                preds = self._m_model.extractFeature(im)
                preds = np.array([preds[0].cpu().numpy()])
                self._faiss_index.add(preds) #add the representation to index
                self._im_indices.append(f)   #store the image name to find it later on
        print("Indexing successfully!")

    def retrieving(self, img_input_path):
        with torch.no_grad():
            for f in os.listdir(img_input_path):
                print("------------query image: ", f)
                im = Image.open(os.path.join(img_input_path,f))
                im = im.resize((224,224))
                im = torch.tensor([self._m_data_transform.val_transforms(im).numpy()]).to(device=self._device)
            
                test_embed = self._m_model.extractFeature(im).cpu().numpy()
                _, I = self._faiss_index.search(test_embed, 5)
                print("Retrieved Image: {}".format(self._im_indices[I[0][0]]))
    
##############################################################################################
# global variable 
flower_transforms = MyTransforms()
PATH_TRAIN = "..\dataset\\102flowers_categorized\dataset\\train"
PATH_VALID = "..\dataset\\102flowers_categorized\dataset\\valid"
PATH_TEST = "..\dataset\\102flowers_categorized\dataset\\test"

# Datasets and Dataloaders
flower_train_data = TripletData(PATH_TRAIN, flower_transforms.train_transforms)
flower_val_data = TripletData(PATH_VALID, flower_transforms.val_transforms)

flower_train_loader = torch.utils.data.DataLoader(dataset = flower_train_data, batch_size=32, shuffle=True, num_workers=2)
flower_val_loader = torch.utils.data.DataLoader(dataset = flower_val_data, batch_size=32, shuffle=False, num_workers=2)

flower_search = FlowerImageSearch()

flower_search.trainDescriptor(flower_train_loader)
flower_search.indexing(PATH_TRAIN)
flower_search.retrieving(PATH_TEST)