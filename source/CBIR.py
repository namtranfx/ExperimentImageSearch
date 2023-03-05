import os

import glob
import faiss 
import torch
from PIL import Image
import numpy as np


from source.data_handler import MyTransforms
from source.features import ResDeepFeature

class FlowerImageSearch:
    def __init__(self) -> None:
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._m_data_transform = MyTransforms()
        self._m_model = ResDeepFeature(self._device)
        self._faiss_index = faiss.IndexFlatL2(1000)   # build the index
        self._im_indices = []

    def trainDescriptor(self, data_loader):
        self._m_model.trainDescriptor(data_loader)
    def saveDescriptorWeight(self, model_path):
        self._m_model.saveDescriptor(model_path)
    def loadDescriptorWeight(self, model_path):
        self._m_model.loadDescriptor(model_path)


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
    