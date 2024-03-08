from configparser import ConfigParser
import ast
import argparse

# Import source
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import os
import sys

from source.data_handler import MyTransform, CaltechDataset, CifarDataset, Oxford102Flower, CustomCocoDataset, DatabaseImage
from source.CBIR import CBIR
from source.features import *
from source.index import *
from source.ultis import to_rgb
########################################################
# Analyse configuration
config = ConfigParser()
config.read('config.ini')
# Analyse Command Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dbname', action="store", dest='dbname')
parser.add_argument('--dbpath', action="store", dest='dbpath')
parser.add_argument('--querypath', action="store", dest='querypath')
parser.add_argument('--index', action='store_true')
parser.add_argument('--retrieve', action='store_true')
parser.add_argument('--usehash', action='store_true')
parser.add_argument('--timeprior', action='store_true')
parser.add_argument('--evalmode', action='store_true')
parser.add_argument('--multitest', action='store_true')

args = parser.parse_args()
# ##################### DATASET ##############################
root_path = ast.literal_eval(config.get('database', 'data_root_folder'))
is_local = ast.literal_eval(config.get('platform', 'is_local'))
database_name = ast.literal_eval(config.get('database', 'database_name'))

transform_img = MyTransform()
mydataloader = []
if args.evalmode:
    for name in database_name:
        if name == "caltech101": 
            # CALTECH-101 DATASET
            PATH_CALTECH101 = os.path.join(root_path, "caltech-101\\101_ObjectCategories" if is_local else "caltech-101/101_ObjectCategories" )

            caltech101ds = CaltechDataset(PATH_CALTECH101, transform_img)
            caltech_train_indices, caltech_test_indices = train_test_split(range(len(caltech101ds)),stratify=caltech101ds.getLabels(), test_size=0.2)
            caltech101_train = torch.utils.data.Subset(caltech101ds, caltech_train_indices)
            caltech101_test = torch.utils.data.Subset(caltech101ds, caltech_test_indices)
            mydataloader.append([DataLoader(caltech101_train, batch_size=1), DataLoader(caltech101_test, batch_size=1)])
        elif name == "cifar10": 
            # CIFAR-10 DATASET
            PATH_CIFAR10 = os.path.join(root_path, "cifar-10\\train" if is_local else "cifar-10/train")
            csv_label_path = os.path.join(root_path, "cifar-10\\trainLabels.csv" if is_local else "cifar-10/trainLabels.csv")

            cifar10ds = CifarDataset(PATH_CIFAR10, csv_label_path, transform_img)
            cifar10_train_indices, cifar10_test_indices = train_test_split(range(len(cifar10ds)),stratify=cifar10ds.getLabels(), test_size=0.2)
            cifar10_train = torch.utils.data.Subset(cifar10ds, cifar10_train_indices)
            cifar10_test = torch.utils.data.Subset(cifar10ds, cifar10_test_indices)
            mydataloader.append([DataLoader(cifar10_train, batch_size=1), DataLoader(cifar10_test, batch_size=1)])
        elif name == "oxford102flower": 
            # OXFORD-102-FLOWER DATASET
            PATH_OXFORD102FLOWERS_TRAIN = os.path.join(root_path, "102flowers_categorized\\dataset\\train" if is_local else "102flowers_categorized/dataset/train")
            PATH_OXFORD102FLOWERS_TEST = os.path.join(root_path, "102flowers_categorized\\dataset\\valid" if is_local else "102flowers_categorized/dataset/valid")

            oxford102flower_train = Oxford102Flower(PATH_OXFORD102FLOWERS_TRAIN, transform_img)
            oxford102flower_test = Oxford102Flower(PATH_OXFORD102FLOWERS_TEST, transform_img)
            mydataloader.append([DataLoader(oxford102flower_train, batch_size=1), DataLoader(oxford102flower_test, batch_size=1)])
        elif name == "coco2017": 
            # MS-COCO 2017 DATASET
            dataDir = os.path.join(root_path, 'coco2017')

            dataType_val = 'val2017'
            dataType_train = 'train2017'

            annFile_train = f'{dataDir}/annotations/instances_{dataType_train}.json'
            annFile_val = f'{dataDir}/annotations/instances_{dataType_val}.json'

            coco_train = CustomCocoDataset(root=f'{dataDir}/{dataType_train}', annFile=annFile_train, transform=transform_img)
            coco_val = CustomCocoDataset(root=f'{dataDir}/{dataType_val}', annFile=annFile_val, transform=transform_img)

            mydataloader.append([DataLoader(coco_train, batch_size=1), DataLoader(coco_val, batch_size=1)])

        else: print("[ERROR]: Check your evaluated dataset name!") 
else:
    test_db = DatabaseImage(args.dbpath, transform_img)
    query_img = DatabaseImage(args.querypath, transform_img)

    mydataloader.append([DataLoader(test_db, batch_size=1), DataLoader(query_img, batch_size=1)])
    database_name = [args.dbname]

# #################### BACKBONE_INSTANCE ################################
RawIndex_bitdepth = [0] # No changing
# Get class name list from config file
if args.multitest or args.evalmode:
    model_names = ast.literal_eval(config.get('backbone', 'model_names'))
    # Get feature_dim array from model_fsize in config file
    feature_dim = ast.literal_eval(config.get('backbone', 'model_fsize'))
    # Get config of indexing
    FaissLSH_bitdepth = ast.literal_eval(config.get('indexing', 'FaissLSH_bitdepth'))
    # And get type of indexing method
    index_type = ast.literal_eval(config.get('indexing', 'index_type')) # IMPORTANT PARAM ===============
else:
    # Determine backbone
    if args.timeprior:
        model_names = ['MobileNetV3Feature']
        feature_dim = [576]
    else:
        model_names = ['tinyvit']
        feature_dim = [576]
    # Determine index type
    if args.usehash:
        index_type = [1]
    else:
        index_type = [0]
    # Determine index config
    FaissLSH_bitdepth = [512]    
# Create backbone model list
BackBoneInstance = []
for model_name in model_names:
    ModelClass = globals()[model_name]
    obj = ModelClass()
    BackBoneInstance.append(obj)

# ###################### INDEX SYSTEM ###################################
bitdepth_config = [
    RawIndex_bitdepth,
    FaissLSH_bitdepth
]
index_creator_config = [
    FaissRawIndex,
    FaissLSHIndex
]
# ####################### AUTO TEST CASE PREPARATION ########################
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
                perform_index_db.append(args.index)
                perform_eval_db.append(args.retrieve)
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
