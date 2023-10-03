import os
import time
from datetime import date

import random
import glob
from tqdm import tqdm
import faiss 
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from faiss import write_index, read_index

from source.metrics import AP
from source.index import MyIndex
from source.features import FeatureDescriptor

def list2string(mylist : list):
    final_str = ""
    final_list = []
    final_list.append(mylist[0])
    for idx in range(1, len(mylist), 1):
        is_save = True
        for dst in final_list:
            if mylist[idx] == dst: 
                is_save = False
                break
        if is_save == True:
            final_list.append(mylist[idx])
    # print("All raw label: ", final_list)

    for i in range(0, len(final_list)):
        if len(final_str) + len(final_list[i]) >= 255: break
        final_str = final_str +  ("_" if i != 0 else "") + str(final_list[i]).strip("'(),")
    return final_str.strip(" ")


def showRetrievalResult(query, query_label, result, result_label, curr_AP, id, meta):
    """
    Show retrieval result.

    Parameters:
        query (PIL.Image): query image
        query_label: label of query (just in evaluation mode)
        result: list of retrieved image file path
        result_label: list of label of retrieved image
        curr_AP: Average Precision of query result
        id: index of query image
        meta: metadata for naming the result img file
    """
    figure_cfg = dict(
        left  = 0.125,  # the left side of the subplots of the figure
        right = 0.9,    # the right side of the subplots of the figure
        bottom = 0.1,   # the bottom of the subplots of the figure
        top = 0.9,      # the top of the subplots of the figure
        wspace = 0.2,   # the amount of width reserved for blank space between subplots
        hspace = 0.2   # the amount of height reserved for white space between subplots
    )

    numcol = 3
    numrow = 2
    # Reprocessing query image (from Torch tensor to RGB array image)
    
    image_np = np.array(query)
    # print("image shape before save: ", image_np.shape)
    image_np = np.squeeze(image_np) # Remove channel has length 1
    image_np = np.transpose(image_np, (1,2,0)) # Transpose to move color channels to the end
    # Show retrieved result
    fig, axs = plt.subplots(numrow, numcol)
    #fig.subplots_adjust(left=figure_cfg["left"], bottom=figure_cfg["bottom"], right=figure_cfg["right"], top=figure_cfg["top"], wspace=figure_cfg["wspace"], hspace=figure_cfg["hspace"])
    fig.tight_layout()
    fig.suptitle("Retrieval AP = " + str(curr_AP))
    idx_img = 0
    axs[0,0].imshow(image_np)
    axs[0,0].set_title(query_label)
    for i in range(numrow):
        for j in range(numcol):
            if (i * numcol + j) == 0: continue
            axs[i,j].imshow(imread(result[idx_img][0]))
            axs[i,j].set_title(list2string(result_label[idx_img]) if len(result_label[idx_img]) != 0 else "<empty>")
            idx_img = idx_img + 1
    #print("Retrieved Image: {}".format(result[0]))
    plt.savefig("." + os.sep + "result" + os.sep + meta[0] + "_by_" + meta[1] + "_with_" + str(id) + ".png", bbox_inches='tight')
    plt.close()

def saveRetrievalResult(query, query_label, result, result_label, curr_AP, id, meta):
    """
    Show retrieval result.

    Parameters:
        query (PIL.Image): query image
        query_label: label of query (just in evaluation mode)
        result: list of retrieved image file path
        result_label: list of label of retrieved image
        curr_AP: Average Precision of query result
        id: index of query image
        meta: metadata for naming the result img file
    """
    # print("shape of query = ", query.shape)
    # print("type of query = ", type(query))
    newpath = "." + os.sep + "result" + os.sep + meta[0] + "_by_" + meta[1] + "_with_" + str(id)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    transform = T.ToPILImage()
    imgx = transform(query[0])
    # print("error path: ",newpath+"\\Query_mAP_" + str(curr_AP) + "_" + list2string(query_label)+".jpg")
    imgx = imgx.save(newpath+ os.sep + "Query-mAP-" + str(curr_AP) + "-" + list2string(query_label)+".jpg")
    for idx in range(0, len(result), 1):
        imgx = Image.open(result[idx][0], mode="r")
        # print("error path: ", newpath+"\\Result_at_rank" + str(idx+1) + "_" + list2string(result_label[idx] if len(result_label[idx]) != 0 else "<empty>")+".jpg")

        imgx = imgx.save(newpath+ os.sep + "Result-at-rank" + str(idx+1) + "-" + list2string(result_label[idx] if len(result_label[idx]) != 0 else "empty")+".jpg")
        
def savelog(filename, content):
    curr_date = str(date.today())
    curr_time = time.strftime("%Hh%Mm%Ss", time.localtime())
    log_file = open("." + os.sep + "log" + os.sep + curr_date + "_" + curr_time + "_" + filename + ".txt", "w")
    log_file.write(content)

    log_file.close()

class CBIR:
    # metadata structure: [dataset_name, model_name]
    def __init__(self, model : FeatureDescriptor, index_sys : MyIndex, metadata, transfer_index = False, evalmode = False) -> None:
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
        self.INDEX_PATH = "." + os.sep + "index" + os.sep + str(metadata[0]) + "_" + str(metadata[1]) + ".index"
        self.DB_IMGPATH = "." + os.sep + "index" + os.sep + str(metadata[0]) + "_imgpath.npy"
        self.DB_LABELS_PATH = "." + os.sep + "index" + os.sep + str(metadata[0]) + "_labels.npy"
    def loadDB(self):
        if self.is_ready == False:
            self.my_index.loadIndex(self.INDEX_PATH)
            self._im_indices = np.load(self.DB_IMGPATH) if self.only_evalmode == False else []
            self._labels = np.load(self.DB_LABELS_PATH, allow_pickle=True)
            print("Database loaded!")
        else:
            print("System has already loaded before!")
    def indexDB(self, dataloader):
        # save log
        log = ""
        log = log + "Indexing " + str(len(dataloader)) + " images of database "+ self.metadata[0] + " using feature " + self.metadata[1] + " ...\n"
        
        print("Indexing", len(dataloader), "images of database", self.metadata[0], "using feature", self.metadata[1], "...")
        sum_time = 0
        sum_extract_time = 0
        sum_index_time = 0
        min_extract_time = 2**31
        min_index_time = 2**31
        
        # Change the model mode into evaluation
        self._m_model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                start_time = time.time()
                if self.only_evalmode == True: 
                    if len(batch) != 2: print("Print the batch: ", batch)
                    img, label = batch
                    img = img.to(self._m_model._device)
                else: 
                    img, label, filepath = batch
                    img = img.to(self._m_model._device)
                feature = self._m_model.extractFeature(img)
                start_index = time.time()
                if min_extract_time > (start_index - start_time): min_extract_time = start_index - start_time
                sum_extract_time = sum_extract_time + (start_index - start_time)
                feature = np.array([feature[0].cpu().numpy()])
                # print("[CBIR.py]: shape of feature before indexing: ", feature.shape)
                self.my_index.add(feature) #add the representation to index
                end_time = time.time()
                if min_index_time > (end_time - start_index): min_index_time = end_time - start_index
                sum_index_time = sum_index_time + (end_time - start_index)
                if self.only_evalmode == False: self._im_indices.append(filepath)   #store the image name to find it later on
                self._labels.append(label)  # Store the label of image above
                end_time = time.time()
                sum_time = sum_time + (end_time - start_time)
        log = log + "Total indexing database progress time: " + str(sum_time) + "s\n"
        log = log + "Average time of index a photo: " + str(sum_time*1000/len(dataloader)) + "ms\n"
        log = log + "------------------------------------------------\n"
        log = log + "Total feature extracting time: " + str(sum_extract_time) + "s\n"
        log = log + "Average time of feature extraction: " + str(sum_extract_time*1000/len(dataloader)) + "ms\n"
        log = log + "Minimum time of feature extraction: " + str(min_extract_time*1000) + "ms\n"
        log = log + "------------------------------------------------\n"
        log = log + "Total index time: " + str(sum_index_time) + "s\n"
        log = log + "Average time of index the feature: " + str(sum_index_time*1000/len(dataloader)) + "ms\n"
        log = log + "Minimum time of index the feature: " + str(min_index_time*1000) + "ms\n"
        log = log + "------------------------------------------------\n"
        savelog(filename="IndexDB_" + self.metadata[0] + "_" + self.metadata[1], content=log)

        print("Indexing successfully in total ", sum_time, "s")
        
        self.is_ready = True
        self.my_index.saveIndex(self.INDEX_PATH)
        if self.only_evalmode == False:
            self._im_indices = np.array(self._im_indices)
            np.save(self.DB_IMGPATH, self._im_indices)
        self._labels = np.array(self._labels)
        np.save(self.DB_LABELS_PATH, self._labels)
        print("Index Saved!")
    
    def retrieve(self, im, k_top = 5):
        """
        retval: tuple[filepaths, labels]
        """
        test_embed = None
        with torch.no_grad():
            test_embed = self._m_model.extractFeature(im)
        image_indices_retrieved = self.my_index.search(test_embed.cpu().numpy(), k_top)
        if self.only_evalmode == True: return self._labels[image_indices_retrieved]
        if self.transfer_index == True: return None, self._labels[image_indices_retrieved]
        return self._im_indices[image_indices_retrieved], self._labels[image_indices_retrieved]
    def evalOnSingleQuery(self, img_input_path, k_top=5):
        """
        Retrieve from a folder of random images
        Parameters:
            img_input_path: path of folder that contains query image request
            k_top: number of retrieval elements
        """
        self.loadDB()
        # self._m_model.loadDescriptor()
        print("Start querying database", self.metadata[0], "using feature", self.metadata[1], "with k =", k_top, "...")
        all_files_and_dirs = os.listdir(img_input_path)
        files_only = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(img_input_path, f))]
        with torch.no_grad():

            for img_filename in files_only:
                img_path = os.path.join(img_input_path, img_filename)
                print("---> Querying image: ", img_path)

                img = Image.open(img_path).convert('RGB')
                #print("img shape=", img.shape())
                query_tensor = self._m_model._m_data_transform.process(img)
                query_tensor = query_tensor.unsqueeze(0)
                # Process imagepath to get real_label
                real_label = []
                imgpath_list = img_filename.split("-")
                tag = imgpath_list[-1]
                for i in range(3 if len(imgpath_list) > 3 else len(imgpath_list) - 2, len(imgpath_list) - 1,1):
                    real_label.append("('" + imgpath_list[i] + "',)")
                # print("Real label: ", real_label)

                # Retrieve in index table
                retrieved_imgpath, retrieved_labels = self.retrieve(query_tensor, k_top=k_top)
                
                curr_AP = AP(real_label, retrieved_labels, k_top)

                showRetrievalResult(query_tensor, real_label, retrieved_imgpath, retrieved_labels, curr_AP, tag, self.metadata)
                saveRetrievalResult(query_tensor, real_label, retrieved_imgpath, retrieved_labels, curr_AP, tag, self.metadata)
                
    def evalRetrieval(self, dataloader, k_top, list_demo:list = []):
        """
        Evaluation system on a validation dataset which have label
        Parameter:
            dataloader: Dataloader store the valid set
            k_top: number of retrieval elements
            list_demo: list that indicate the index of image wanna using as single query
        """
        if len(list_demo) == 0:
            number_demo = 5
            list_demo = random.sample(range(1,len(dataloader.dataset)), number_demo)
        self.loadDB()
        # self._m_model.loadDescriptor()
        log = "Start evaluating database " + self.metadata[0] + " using feature " + self.metadata[1] + " with k = " + str(k_top) + "...\n"
        print("Start evaluating database", self.metadata[0], "using feature", self.metadata[1], "with k =", k_top, "...")
        sumAP = 0
        num_query = 0
        sum_time = 0
        min_time = 2**31
        min_AP = 100
        minAP_data = None
        max_AP = 0
        maxAP_data = None

        self._m_model.eval()
        for batch in tqdm(dataloader):
            #------------------------------------
            start_query_time = time.time()
            if self.only_evalmode == False:
                img, real_label, filepath = batch
                img = img.to(self._m_model._device)
                retrieved_imgpath, retrieved_labels = self.retrieve(img, k_top=k_top)
            else:
                img, real_label = batch
                img = img.to(self._m_model._device)
                retrieved_labels = self.retrieve(img, k_top=k_top)
            end_query_time = time.time()
            sum_time = sum_time + (end_query_time - start_query_time)
            if min_time > (end_query_time - start_query_time): 
                min_time = end_query_time - start_query_time
            

            curr_AP = AP(real_label, retrieved_labels, k_top)
            if min_AP > curr_AP*100: 
                min_AP = curr_AP * 100
                del minAP_data
                minAP_data = list([img.cpu(), real_label, retrieved_imgpath, retrieved_labels, num_query])
            if max_AP < curr_AP*100: 
                max_AP = curr_AP * 100
                del maxAP_data
                maxAP_data = list([img.cpu(), real_label, retrieved_imgpath, retrieved_labels, num_query])

            sumAP = sumAP + curr_AP

            for item in list_demo:
                if num_query == item:
                    showRetrievalResult(img.cpu(), real_label, retrieved_imgpath, retrieved_labels, curr_AP, num_query, self.metadata)
                    saveRetrievalResult(img.cpu(), real_label, retrieved_imgpath, retrieved_labels, curr_AP, num_query, self.metadata)
            num_query = num_query + 1
        
        # Save best and worst retrieval result
        saveRetrievalResult(minAP_data[0], minAP_data[1], minAP_data[2], minAP_data[3], min_AP, str(minAP_data[4]) + "WORST_RESULT", self.metadata)
        showRetrievalResult(minAP_data[0], minAP_data[1], minAP_data[2], minAP_data[3], min_AP, str(minAP_data[4]) + "WORST_RESULT", self.metadata)
        
        saveRetrievalResult(maxAP_data[0], maxAP_data[1], maxAP_data[2], maxAP_data[3], max_AP, str(maxAP_data[4]) + "BEST_RESULT", self.metadata)
        showRetrievalResult(maxAP_data[0], maxAP_data[1], maxAP_data[2], maxAP_data[3], max_AP, str(maxAP_data[4]) + "BEST_RESULT", self.metadata)
        
        if num_query == 0: print("No query image found!")
        else:
            log = log + "Performance of system: mAP = " + str(sumAP * 100/num_query) + "%\n"
            log = log + "Max Performance of system at a single query: MaxAP = " + str(max_AP) + "%\n"
            log = log + "------------------------------------------------\n"
            log = log + "Total query progress time: " + str(sum_time) + "s\n"
            log = log + "Average time of a query request: " + str(sum_time*1000/num_query) + "ms\n"
            log = log + "Minimum query time: " + str(min_time*1000) + "ms\n"
            
            print("mAP =", sumAP/num_query, "(on", num_query, "queries)")
            print("total query time:", sum_time, "s", "(", sum_time*1000/num_query, "ms each query)")
            savelog(filename="Retrieval_" + self.metadata[0] + "_" + self.metadata[1], content=log)
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



    