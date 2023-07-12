import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from source.data_handler import MyTransform_norm, TripletData
from source.data_handler import InriaHolidayDataset, CorelDataset, CaltechDataset, OxfordDataset
from source.CBIR import FlowerImageSearch, CBIR
from source.features import *




# global variable 
flower_transforms = MyTransform_norm()
PATH_TRAIN = "..\dataset\\102flowers_categorized\dataset\\train\\"
PATH_VALID = "..\dataset\\102flowers_categorized\dataset\\valid\\"
PATH_TEST = "..\dataset\\102flowers_categorized\dataset\\smalltest\\"


perform_index = False
eval_mode = False

# Datasets and Dataloaders
# flower_train_data = TripletData(PATH_TRAIN, flower_transforms.train_transforms)
# flower_val_data = TripletData(PATH_VALID, flower_transforms.val_transforms)

# flower_train_loader = torch.utils.data.DataLoader(dataset = flower_train_data, batch_size=32, shuffle=True, num_workers=2)
# flower_val_loader = torch.utils.data.DataLoader(dataset = flower_val_data, batch_size=32, shuffle=False, num_workers=2)

flowersearch = [ FlowerImageSearch(Resnet18_custom_best(), index_size=512, metadata="best_resnet18")]#,
                # FlowerImageSearch(Resnet18Descriptor(), index_size=512, metadata="resnet18"),
                # FlowerImageSearch(Resnet34Descriptor(), index_size=512, metadata="resnet34"),
                # FlowerImageSearch(Resnet50Descriptor(), index_size=2048, metadata="resnet50")]

k_top = [5,7,9,11]
#flower_search.trainDescriptor(flower_train_loader)
#flower_search.saveDescriptorWeight()

if perform_index == True:
    for isys in flowersearch:
        print("----------------------------")
        isys.indexing(PATH_TRAIN)
if eval_mode == False:
    for isys in flowersearch:
        isys.retrieving(PATH_TEST)
else:
    for isys in flowersearch:
        print("----------------------------")
        for i in k_top:
            isys.evalOnDataset(PATH_VALID, k_top=i)


