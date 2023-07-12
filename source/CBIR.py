import os
import time

import glob
from tqdm import tqdm
import faiss 
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from faiss import write_index, read_index

from source.metrics import AP
from source.index import MyIndex

class CBIR:
    # metadata structure: [dataset_name, model_name]
    def __init__(self, model, index_sys : MyIndex, metadata, transfer_index = False, evalmode = False) -> None:
        self.metadata = metadata
        # Feature descriptor
        self._m_model = model
        # Indexing system
        self.my_index = index_sys
        self._im_indices = []
        self._labels = []
        # sys config
        self.is_ready = False
        self.only_evalmode = evalmode
        self.transfer_index = transfer_index

        # Path for database
        self.INDEX_PATH = ".\index\\" + str(metadata[0]) + "_" + str(metadata[1]) + ".index"
        self.DB_IMGPATH = ".\index\\" + str(metadata[0]) + "_imgpath.npy"
        self.DB_LABELS_PATH = ".\index\\" + str(metadata[0]) + "_labels.npy"
    def loadDB(self):
        if self.is_ready == False:
            self.my_index.loadIndex(self.INDEX_PATH)
            self._im_indices = np.load(self.DB_IMGPATH) if self.only_evalmode == False else []
            self._labels = np.load(self.DB_LABELS_PATH, allow_pickle=True)
            print("Database loaded!")
        else:
            print("System has already loaded before!")
    def indexDB(self, dataloader):
        print("Indexing", len(dataloader), "images of database", self.metadata[0], "using feature", self.metadata[1], "...")
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if self.only_evalmode == True: 
                    if len(batch) != 2: print("Print the batch: ", batch)
                    img, label = batch
                else: img, label, filepath = batch
                feature = self._m_model.extractFeature(img)
                feature = np.array([feature[0].cpu().numpy()])
                # print("[CBIR.py]: shape of feature before indexing: ", feature.shape)
                self.my_index.add(feature) #add the representation to index
                if self.only_evalmode == False: self._im_indices.append(filepath)   #store the image name to find it later on
                self._labels.append(label)  # Store the label of image above
        end_time = time.time()
        print("Indexing successfully in total ", end_time - start_time, "s")
        self.is_ready = True
        self.my_index.saveIndex(self.INDEX_PATH)
        if self.only_evalmode == False:
            self._im_indices = np.array(self._im_indices)
            np.save(self.DB_IMGPATH, self._im_indices)
        self._labels = np.array(self._labels)
        np.save(self.DB_LABELS_PATH, self._labels)
        print("Index Saved!")
    # retval: tuple[filepaths, labels]
    def retrieve(self, im, k_top = 5):
        test_embed = None
        with torch.no_grad():
            test_embed = self._m_model.extractFeature(im).cpu().numpy()
        image_indices_retrieved = self.my_index.search(test_embed, k_top)
        if self.only_evalmode == True: return self._labels[image_indices_retrieved]
        if self.transfer_index == True: return None, self._labels[image_indices_retrieved]
        return self._im_indices[image_indices_retrieved], self._labels[image_indices_retrieved]
    def testRetrieval(self, img_input_path):
        self.loadDB()
        # self._m_model.loadDescriptor()
        numcol = 3
        numrow = 2
        
        with torch.no_grad():
            for f in os.listdir(img_input_path):
                print("------------query image: ", f)
                im = Image.open(os.path.join(img_input_path,f))
                result, _ = self.retrieve(im, 5)

                # Show retrieved result
                fig, axs = plt.subplots(numrow, numcol)
                fig.suptitle("Retrieved Result")
                idx_img = 0
                axs[0,0].imshow(imread(os.path.join(img_input_path,f)))
                for i in range(numrow):
                    for j in range(numcol):
                        if (i * numcol + j) == 0: continue
                        axs[i,j].imshow(imread(result[idx_img]))
                        idx_img = idx_img + 1
                print("Retrieved Image: {}".format(result[0]))
                plt.show()
    def evalRetrieval(self, dataloader, k_top):
        self.loadDB()
        # self._m_model.loadDescriptor()
        print("Start evaluating database", self.metadata[0], "using feature", self.metadata[1], "with k =", k_top, "...")
        sumAP = 0
        num_query = 0
        start_time = time.time()
        for batch in tqdm(dataloader):
            if self.only_evalmode == False:
                img, real_label, filepath = batch
                retrieved_imgpath, retrieved_labels = self.retrieve(img, k_top=k_top)
            else:
                img, real_label = batch
                retrieved_labels = self.retrieve(img, k_top=k_top)
            num_query = num_query + 1

            curr_AP = AP(real_label, retrieved_labels, k_top)
            sumAP = sumAP + curr_AP
        end_time = time.time()
        print("mAP =", sumAP/num_query, "(on", num_query, "queries)")
        print("total query time:", (end_time-start_time), "s", "(", (end_time-start_time) * 10**3/num_query, "ms each query)")
        
############# IMAGE RETRIEVAL FOR 192FLOWER DATASET #############################
class FlowerImageSearch:
    def __init__(self, model, index_size, metadata) -> None:
        
        # self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metadata = metadata
        # Feature descriptor
        self._m_model = model

        # Indexing system
        self._faiss_index = faiss.IndexFlatL2(index_size)   # build the index
        self._im_indices = []
        self.is_ready = False

        self.INDEX_PATH = ".\index\\102flower_" + str(metadata) + ".index"
        self.DB_IMGPATH = ".\index\\db_imgpath.npy"

    def trainDescriptor(self, data_loader):
        self._m_model.trainDescriptor(data_loader)
    def saveDescriptorWeight(self):
        self._m_model.saveDescriptor()
    # def loadDescriptorWeight(self):
    #     self._m_model.loadDescriptor()

    def loadDB(self):
        if self.is_ready == False:
            self._faiss_index = read_index(self.INDEX_PATH)
            self._im_indices = np.load(self.DB_IMGPATH)
            print("Database loaded!")
        else:
            print("System has already loaded before!")

    def indexing(self, db_path):
        # self._m_model.loadDescriptor()
        print("Indexing your database using", self.metadata, "...")
        start_time = time.time()
        with torch.no_grad():
            for f in glob.glob(os.path.join(db_path, '*/*')):
                im = Image.open(f)
                
                preds = self._m_model.extractFeature(im)
                preds = np.array([preds[0].cpu().numpy()])
                
                self._faiss_index.add(preds) #add the representation to index
                self._im_indices.append(f)   #store the image name to find it later on
        end_time = time.time()
        print("Indexing successfully in total ", end_time - start_time, "s")
        self.is_ready = True
        write_index(self._faiss_index, self.INDEX_PATH)
        self._im_indices = np.array(self._im_indices)
        np.save(self.DB_IMGPATH, self._im_indices)
        print("Index Saved!")
    def retrieve(self, im, k_top = 5):
        test_embed = self._m_model.extractFeature(im).detach().numpy()
        _, I = self._faiss_index.search(test_embed, k_top)
        return self._im_indices[I[0][:]]
    def retrieving(self, img_input_path):
        self.loadDB()
        # self._m_model.loadDescriptor()
            
        numcol = 3
        numrow = 2
        
        with torch.no_grad():
            for f in os.listdir(img_input_path):
                print("------------query image: ", f)
                im = Image.open(os.path.join(img_input_path,f))
                result = self.retrieve(im, 5)

                # Show retrieved result
                fig, axs = plt.subplots(numrow, numcol)
                fig.suptitle("Retrieved Result")
                idx_img = 0
                axs[0,0].imshow(imread(os.path.join(img_input_path,f)))
                for i in range(numrow):
                    for j in range(numcol):
                        if (i * numcol + j) == 0: continue
                        axs[i,j].imshow(imread(result[idx_img]))
                        idx_img = idx_img + 1
                print("Retrieved Image: {}".format(result[0]))
                plt.show()
    def evalOnDataset(self, eval_path, k_top):
        self.loadDB()
        # self._m_model.loadDescriptor()
        print("Start evaluating using", self.metadata, "with k =", k_top, "...")
        sumAP = 0
        num_query = 0
        # test_flag = True
        start_time = time.time()
        
        for f in glob.glob(os.path.join(eval_path, '*/*')):
            im = Image.open(f)
            result = self.retrieve(im, k_top=k_top)

            # Compare the result for evaluation
            input_category = f.split("\\")[-2]
            retrieved_category = np.fromiter(map(lambda x:x.split("\\")[-2], result), dtype=np.int32)
            num_query = num_query + 1
            
            #sumAP = sumAP + AP(input_category, retrieved_category, k_top)
            # print("ground_true label: ", input_category)
            # print("retrieval result: ", retrieved_category)

            curr_AP = AP(input_category, retrieved_category, k_top)
            sumAP = sumAP + curr_AP
            
            
            # print("AP = ", curr_AP)
            # print("-------------------------------------")
        end_time = time.time()
        print("mAP =", sumAP/num_query, "(on", num_query, "queries)")
        print("total query time:", (end_time-start_time), "s", "(", (end_time-start_time) * 10**3/num_query, "ms each query)")



    