from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import os
import sys

from source.data_handler import MyTransform, CaltechDataset, CifarDataset, Oxford102Flower, CustomCocoDataset
from source.CBIR import CBIR
from source.features import *
from source.index import *
from source.ultis import to_rgb

############################################################################################
################################# DATASET PATH #############################################
############################################################################################

# PATH_COREL5K = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\Corel-5k\\images"
# PATH_COREL10K = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\Corel-10k"
# PATH_HOLIDAY = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\The INRIA Holidays dataset\\jpg"
# PATH_OXFORD5K = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\oxbuild_images"
# PATH_CALTECH256 = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\caltech-101\\256_ObjectCategories"


# datasetpath = [PATH_COREL5K, PATH_COREL10K, PATH_HOLIDAY, PATH_OXFORD5K, PATH_CALTECH101, PATH_CALTECH256]

############################################################################################
################################# CUSTOM DATASET ###########################################
############################################################################################
transform_img = MyTransform()

# # CALTECH-101 DATASET
# PATH_CALTECH101 = "/kaggle/input/cbir-ds/caltech-101/caltech-101/101_ObjectCategories"
# caltech101ds = CaltechDataset(PATH_CALTECH101, transform_img)
# caltech_train_indices, caltech_test_indices = train_test_split(range(len(caltech101ds)),stratify=caltech101ds.getLabels(), test_size=0.2)
# caltech101_train = torch.utils.data.Subset(caltech101ds, caltech_train_indices)
# caltech101_test = torch.utils.data.Subset(caltech101ds, caltech_test_indices)
# # CIFAR-10 DATASET
# PATH_CIFAR10 = "/kaggle/input/cbir-ds/cifar-10/cifar-10/train"
# csv_label_path = "/kaggle/input/cbir-ds/cifar-10/cifar-10/trainLabels.csv"

# cifar10ds = CifarDataset(PATH_CIFAR10, csv_label_path, transform_img)
# cifar10_train_indices, cifar10_test_indices = train_test_split(range(len(cifar10ds)),stratify=cifar10ds.getLabels(), test_size=0.2)
# cifar10_train = torch.utils.data.Subset(cifar10ds, cifar10_train_indices)
# cifar10_test = torch.utils.data.Subset(cifar10ds, cifar10_test_indices)
# # OXFORD-102-FLOWER DATASET
# PATH_OXFORD102FLOWERS_TRAIN = "/kaggle/input/cbir-ds/102flowers_categorized/102flowers_categorized/dataset/train"
# PATH_OXFORD102FLOWERS_TEST = "/kaggle/input/cbir-ds/102flowers_categorized/102flowers_categorized/dataset/valid"
# oxford102flower_train = Oxford102Flower(PATH_OXFORD102FLOWERS_TRAIN, transform_img)
# oxford102flower_test = Oxford102Flower(PATH_OXFORD102FLOWERS_TEST, transform_img)

# NUS-WIDE DATASET

# MS-COCO 2017 DATASET
dataDir_train = '/kaggle/input/cbir-ds/train2017'
dataDir_val = '/kaggle/input/cbir-ds/val2017'
annFileDir = '/kaggle/input/cbir-ds/annotations_trainval2017'

dataType_val = 'val2017'
dataType_train = 'train2017'

annFile_train = f'{annFileDir}/annotations/instances_{dataType_train}.json'
annFile_val = f'{annFileDir}/annotations/instances_{dataType_val}.json'

coco_train = CustomCocoDataset(root=f'{dataDir_train}/{dataType_train}', annFile=annFile_train, transform=transform_img)
coco_val = CustomCocoDataset(root=f'{dataDir_val}/{dataType_val}', annFile=annFile_val, transform=transform_img)

# =============================================================================================

# inriaHoliday_train = torch.utils.data.Subset(inriaHolidayds, inriaHoliday_train_indices)
# inriaHoliday_test = torch.utils.data.Subset(inriaHolidayds, inriaHoliday_test_indices)

############################################################################################
################################# CUSTOM DATALOADER ########################################
############################################################################################
# transform = torchvision.transforms.Compose(
    
#     [torchvision.transforms.Lambda(lambda image: to_rgb(image)),
#      torchvision.transforms.Resize((224, 224)),
#      torchvision.transforms.ToTensor()#,
#      #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#      ])
# cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=1)#, shuffle=True, num_workers=2)
# cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=1)#, shuffle=False, num_workers=2)
#-----------------------------------------------

# caltech101 = torchvision.datasets.Caltech101(root='./data', download=True, transform=transform)
# caltech101train, caltech101test = torch.utils.data.random_split(caltech101, [int(len(caltech101)*0.8), len(caltech101) - int(len(caltech101)*0.8)])
# caltech101_trainloader = torch.utils.data.DataLoader(caltech101train, batch_size=1)#, shuffle=True, num_workers=2)
# caltech101_testloader = torch.utils.data.DataLoader(caltech101test, batch_size=1)#, shuffle=False, num_workers=2)

#--------------------------------------------------
# flickr30k = torchvision.datasets.Flickr30k(root='./data', transform=transform, ann_file=".\\data\\captions.txt")
# flickr30k_train, flickr30k_test = train_test_split(range(len(flickr30k)), test_size=0.2)
# flickr30k_trainloader = torch.utils.data.DataLoader(flickr30k_train, batch_size=1)#, shuffle=True, num_workers=2)
# flickr30k_testloader = torch.utils.data.DataLoader(flickr30k_test, batch_size=1)#, shuffle=False, num_workers=2)


#-----------------------------------------------

all_dataloader = [
    # [DataLoader(caltech101_train, batch_size=1), DataLoader(caltech101_test, batch_size=1)],
    # [DataLoader(cifar10_train, batch_size=1), DataLoader(cifar10_test, batch_size=1)],
    # [DataLoader(oxford102flower_train, batch_size=1), DataLoader(oxford102flower_test, batch_size=1)],
    [DataLoader(coco_train, batch_size=1), DataLoader(coco_val, batch_size=1)],
    # [flickr30k_trainloader, flickr30k_testloader],
    # [caltech101_trainloader, caltech101_testloader],
    # [cifar_trainloader, cifar_testloader],
    # [DataLoader(oxford5k_train, batch_size=1), DataLoader(oxford5k_test, batch_size=1)],
    # [DataLoader(corel5k_train, batch_size=1), DataLoader(corel5k_test, batch_size=1)],
    # [DataLoader(inriaHoliday_train, batch_size=1), DataLoader(inriaHoliday_test, batch_size=1)]
]
BackBoneInstance = [
    # Resnet18_custom_best(),
    # Resnet18Descriptor(),
    # Resnet50Descriptor(), 
    # MobileNetV3Feature(),
    # MobileNetV3Feature_large(),
    SwinTransformer_default(),
    # tinyvit(),
    # tinyvit_small(),
    # MyEfficientViT()
]

############################################################################################
################################# CBIR INSTANCE ############################################
############################################################################################

#================
# Database system
# Name: ["Caltech101", "Cifar10", "Oxford102Flower", "Coco2017"]
# database_name = ["Caltech101", "Cifar10", "Oxford102Flower"]
database_name = ["Coco2017"]
mydataloader = []
# 0: caltech101
# 1: cifar10
# 2: oxford102flower
# 3: coco2017
database_id = [0] # corresponding to index value of dataloader
for idx in database_id:
    mydataloader.append(all_dataloader[idx])
# Index system
# feature_dim = [512, 2048, 576, 960, 768, 576]
feature_dim = [768] # resnet18, resnet50, mobilenetv3_small, mobilenetv3_large, swin_vit, tiny_vit
# feature_dim = [512] # resnet18, resnet50, mobilenetv3_small, mobilenetv3_large, swin_vit, tiny_vit
RawIndex_bitdepth = [0]
FaissLSH_bitdepth = [1024, 2048]
CustomLSH_bitdepth = [1, 2, 3, 4, 5, 6, 7, 8]
bitdepth_config = [
    RawIndex_bitdepth,
    FaissLSH_bitdepth,
    CustomLSH_bitdepth
]
index_creator_config = [
    FaissRawIndex,
    FaissLSHIndex,
    CustomLSHIndex
]
# Index instance creator
# FaissRawIndex: 0
# FaissLSHIndex: 1
# CustomLSHIndex: 2
index_type = [1] # IMPORTANT PARAM ==================================================
Index_instances = []
for index_type_id in range(0, len(index_type), 1):
    index_type_list = []
    for dim in feature_dim:
        index_per_backbone = []
        for bitdep in bitdepth_config[index_type[index_type_id]]:
            if bitdep == 0:
                index_per_backbone.append(index_creator_config[index_type[index_type_id]](dim))
            else:
                index_per_backbone.append(index_creator_config[index_type[index_type_id]](dim, bitdep))
        index_type_list.append(index_per_backbone)
    Index_instances.append(index_type_list)
# Metadata Creator
metadata = []
for index_type_id in range(0, len(index_type), 1):
    metadata_type_level = []
    for db_name in database_name:
        metadata_db_level = []
        for backbone_id in range(0, len(BackBoneInstance), 1):
            metadata_per_backbone = []
            for bitdep_id in range(0, len(bitdepth_config[index_type[index_type_id]]), 1):
                metadata_per_backbone.append([db_name, 
                                              type(BackBoneInstance[backbone_id]).__name__ + 
                                              "_" + 
                                              type(Index_instances[index_type_id][backbone_id][bitdep_id]).__name__ + 
                                              "_" + 
                                              str(bitdepth_config[index_type[index_type_id]][bitdep_id]) + 
                                              "_bits"])
            metadata_db_level.append(metadata_per_backbone)
        metadata_type_level.append(metadata_db_level)
    metadata.append(metadata_type_level)

# CBIR instances Creator
TestSearch = []
for index_type_id in range(0, len(index_type), 1):
    TestSearch_type_level = []
    for db_i in range(0, len(database_name), 1):
        TestSearch_db_level = []
        for backbone_id in range(0, len(BackBoneInstance), 1):
            for bitdep_id in range(0, len(bitdepth_config[index_type[index_type_id]]), 1):
                TestSearch_db_level.append(CBIR(BackBoneInstance[backbone_id], 
                                                Index_instances[index_type_id][backbone_id][bitdep_id],
                                                metadata[index_type_id][db_i][backbone_id][bitdep_id]))
        TestSearch_type_level.append(TestSearch_db_level)
    TestSearch.append(TestSearch_type_level)


# Control Variable
Control_for_type = []
for index_type_id in range(0, len(index_type), 1):
    perform_index = []
    perform_eval = []
    for db_i in range(0, len(database_name), 1):
        perform_index_db = []
        perform_eval_db = []
        for backbone_id in range(0, len(BackBoneInstance), 1):
            for bitdep_id in range(0, len(bitdepth_config[index_type[index_type_id]]), 1):
                perform_index_db.append(True)
                perform_eval_db.append(True)
        perform_index.append(perform_index_db)
        perform_eval.append(perform_eval_db)
    Control_for_type.append([perform_index, perform_eval])
    
k_top = [5]
"""
Control Structure
[

 [        db1         ]
[[        db2         ]] # a type of index 
 [        db3         ]
------------------------
 [        db1         ]
[[        db2         ]] # a type of index 
 [        db3         ]

]
"""


############################################################################################
################################# PERFORM INDEXING AND RETRIEVING ##########################
############################################################################################
for index_type_id in range(0, len(index_type), 1):
    for idx_db in range(0, len(TestSearch[index_type_id]), 1):
        print("================================= Database",idx_db + 1, "=================================")
        ii = 0
        for idx_cbir in range(0, len(TestSearch[index_type_id][idx_db]), 1):
            if Control_for_type[index_type_id][0][idx_db][idx_cbir] == True:
                TestSearch[index_type_id][idx_db][idx_cbir].indexDB(mydataloader[idx_db][0])
            # Evaluate phase
            if Control_for_type[index_type_id][1][idx_db][idx_cbir] == True:
                for k in k_top:
                    TestSearch[index_type_id][idx_db][idx_cbir].evalRetrieval(mydataloader[idx_db][1], k)
                ii = ii + 1
            print("-------------------------------------------------------------------------------")
