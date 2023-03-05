import torch


from source.data_handler import MyTransforms, TripletData
from source.CBIR import FlowerImageSearch


# global variable 
flower_transforms = MyTransforms()
PATH_TRAIN = "..\dataset\\102flowers_categorized\dataset\\train\\"
PATH_VALID = "..\dataset\\102flowers_categorized\dataset\\valid\\"
PATH_TEST = "..\dataset\\102flowers_categorized\dataset\\smalltest\\"
model_path = ".\weight\\result_weight_model.pt"

perform_index = False

# Datasets and Dataloaders
flower_train_data = TripletData(PATH_TRAIN, flower_transforms.train_transforms)
flower_val_data = TripletData(PATH_VALID, flower_transforms.val_transforms)

flower_train_loader = torch.utils.data.DataLoader(dataset = flower_train_data, batch_size=32, shuffle=True, num_workers=2)
flower_val_loader = torch.utils.data.DataLoader(dataset = flower_val_data, batch_size=32, shuffle=False, num_workers=2)

flower_search = FlowerImageSearch()


#flower_search.trainDescriptor(flower_train_loader)
#flower_search.saveDescriptorWeight(model_path=model_path)
flower_search.loadDescriptorWeight(model_path=model_path)
if perform_index == True:
    flower_search.indexing(PATH_TRAIN)
    flower_search.retrieving(PATH_TEST)
else:
    flower_search.retrieving(PATH_TEST, 1) # Set to 0 if want to perform index


