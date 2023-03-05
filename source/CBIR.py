import os

import glob
import faiss 
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from faiss import write_index, read_index


from source.data_handler import MyTransforms
from source.features import ResDeepFeature

# VARIABLE
index_path = ".\index\\102flower.index"
db_imgpath = ".\index\\db_imgpath.npy"

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
        write_index(self._faiss_index, index_path)
        np.save(db_imgpath, self._im_indices)
        print("Index Saved!")

    def retrieving(self, img_input_path, load_index = 0):
        if load_index == 1:
            self._faiss_index = read_index(index_path)
            self._im_indices = np.load(db_imgpath)
        numcol = 3
        numrow = 2

        with torch.no_grad():
            for f in os.listdir(img_input_path):
                print("------------query image: ", f)
                im = Image.open(os.path.join(img_input_path,f))
                im = im.resize((224,224))
                im = torch.tensor([self._m_data_transform.val_transforms(im).numpy()]).to(device=self._device)
            
                test_embed = self._m_model.extractFeature(im).cpu().numpy()
                _, I = self._faiss_index.search(test_embed, 5)
                # Show retrieved result
                fig, axs = plt.subplots(numrow, numcol)
                fig.suptitle("Retrieved Result")
                idx_img = 0
                axs[0,0].imshow(imread(os.path.join(img_input_path,f)))
                for i in range(numrow):
                    for j in range(numcol):
                        if (i * numcol + j) == 0: continue
                        axs[i,j].imshow(imread(self._im_indices[I[0][idx_img]]))
                        idx_img = idx_img + 1
                print("Retrieved Image: {}".format(self._im_indices[I[0][0]]))
                plt.show()
    