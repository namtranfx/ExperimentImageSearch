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
from source.metrics import AP


class FlowerImageSearch:
    def __init__(self) -> None:
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._m_data_transform = MyTransforms()
        self._m_model = ResDeepFeature(self._device)
        self._faiss_index = faiss.IndexFlatL2(1000)   # build the index
        self._im_indices = []
        self.is_ready = False

        self.INDEX_PATH = ".\index\\102flower.index"
        self.DB_IMGPATH = ".\index\\db_imgpath.npy"

    def trainDescriptor(self, data_loader):
        self._m_model.trainDescriptor(data_loader)
    def saveDescriptorWeight(self):
        self._m_model.saveDescriptor()
    def loadDescriptorWeight(self):
        self._m_model.loadDescriptor()

    def loadDB(self):
        if self.is_ready == False:
            self._faiss_index = read_index(self.INDEX_PATH)
            self._im_indices = np.load(self.DB_IMGPATH)
            print("Database loaded!")
        else:
            print("System has already loaded before!")

    def indexing(self, db_path):
        self._m_model.loadDescriptor()
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
        self.is_ready = True
        write_index(self._faiss_index, self.INDEX_PATH)
        np.save(self.DB_IMGPATH, self._im_indices)
        print("Index Saved!")
    def retrieve(self, im, k_top):
        im = im.resize((224,224))
        im = torch.tensor([self._m_data_transform.val_transforms(im).numpy()]).to(device=self._device)

        test_embed = self._m_model.extractFeature(im).detach().numpy()
        _, I = self._faiss_index.search(test_embed, k_top)
        return self._im_indices[I[0][:]]
    def retrieving(self, img_input_path):
        self.loadDB()
        self._m_model.loadDescriptor()
            
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
    def evalOnDataset(self, eval_path, k_top):
        self.loadDB()
        self._m_model.loadDescriptor()
        sumAP = 0
        num_query = 0
        for f in glob.glob(os.path.join(eval_path, '*/*')):
            im = Image.open(f)
            im = im.resize((224,224))
            im = torch.tensor([self._m_data_transform.val_transforms(im).numpy()]).to(device=self._device)

            test_embed = self._m_model.extractFeature(im).detach().numpy()
            _, I = self._faiss_index.search(test_embed, k_top)
            input_category = f.split("\\")[-2]
            retrieved_category = np.fromiter(map(lambda x:x.split("\\")[-2], self._im_indices[I[0][:]]), dtype=np.int32)
            #retrieved_category = self._im_indices[I[0][:]][:].split("\\")[-2]
            num_query = num_query + 1
            
            #sumAP = sumAP + AP(input_category, retrieved_category, k_top)
            print("ground_true label: ", input_category)
            print("retrieval result: ", retrieved_category)
            
            curr_AP = AP(input_category, retrieved_category, k_top)
            sumAP = sumAP + curr_AP
            
            
            print("AP = ", curr_AP)
            print("-------------------------------------")
        print("mAP = ", sumAP/num_query)



    