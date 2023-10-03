from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import os

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

# CALTECH-101 DATASET
PATH_CALTECH101 = "\\content\\dataset\\caltech-101\\101_ObjectCategories"
caltech101ds = CaltechDataset(PATH_CALTECH101, transform_img)
caltech_train_indices, caltech_test_indices = train_test_split(range(len(caltech101ds)),stratify=caltech101ds.getLabels(), test_size=0.2)
caltech101_train = torch.utils.data.Subset(caltech101ds, caltech_train_indices)
caltech101_test = torch.utils.data.Subset(caltech101ds, caltech_test_indices)
# CIFAR-10 DATASET
PATH_CIFAR10 = "\\content\\dataset\\cifar-10\\train"

cifar10ds = CifarDataset(PATH_CIFAR10, transform_img)
cifar10_train_indices, cifar10_test_indices = train_test_split(range(len(cifar10ds)),stratify=cifar10ds.getLabels(), test_size=0.2)
cifar10_train = torch.utils.data.Subset(cifar10ds, cifar10_train_indices)
cifar10_test = torch.utils.data.Subset(cifar10ds, cifar10_test_indices)
# OXFORD-102-FLOWER DATASET
PATH_OXFORD102FLOWERS_TRAIN = "\\content\\dataset\\102flowers_categorized\\dataset\\train"
PATH_OXFORD102FLOWERS_TEST = "\\content\\dataset\\102flowers_categorized\\dataset\\valid"
oxford102flower_train = Oxford102Flower(PATH_OXFORD102FLOWERS_TRAIN, transform_img)
oxford102flower_test = Oxford102Flower(PATH_OXFORD102FLOWERS_TEST, transform_img)

# NUS-WIDE DATASET

# MS-COCO 2017 DATASET
dataDir = '/content/dataset'

dataType_val = 'val2017'
dataType_train = 'train2017'

annFile_train = f'{dataDir}/annotations/instances_{dataType_train}.json'
annFile_val = f'{dataDir}/annotations/instances_{dataType_val}.json'

coco_train = CustomCocoDataset(root=f'{dataDir}/{dataType_train}', annFile=annFile_train, transform=transform_img)
coco_val = CustomCocoDataset(root=f'{dataDir}/{dataType_val}', annFile=annFile_val, transform=transform_img)

# # =============================================================================================

# # inriaHoliday_train = torch.utils.data.Subset(inriaHolidayds, inriaHoliday_train_indices)
# # inriaHoliday_test = torch.utils.data.Subset(inriaHolidayds, inriaHoliday_test_indices)

# ############################################################################################
# ################################# CUSTOM DATALOADER ########################################
# ############################################################################################
# # transform = torchvision.transforms.Compose(
    
# #     [torchvision.transforms.Lambda(lambda image: to_rgb(image)),
# #      torchvision.transforms.Resize((224, 224)),
# #      torchvision.transforms.ToTensor()#,
# #      #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# #      ])
# # cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# # cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# # cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=1)#, shuffle=True, num_workers=2)
# # cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=1)#, shuffle=False, num_workers=2)
# #-----------------------------------------------

# # caltech101 = torchvision.datasets.Caltech101(root='./data', download=True, transform=transform)
# # caltech101train, caltech101test = torch.utils.data.random_split(caltech101, [int(len(caltech101)*0.8), len(caltech101) - int(len(caltech101)*0.8)])
# # caltech101_trainloader = torch.utils.data.DataLoader(caltech101train, batch_size=1)#, shuffle=True, num_workers=2)
# # caltech101_testloader = torch.utils.data.DataLoader(caltech101test, batch_size=1)#, shuffle=False, num_workers=2)

# #--------------------------------------------------
# # flickr30k = torchvision.datasets.Flickr30k(root='./data', transform=transform, ann_file=".\\data\\captions.txt")
# # flickr30k_train, flickr30k_test = train_test_split(range(len(flickr30k)), test_size=0.2)
# # flickr30k_trainloader = torch.utils.data.DataLoader(flickr30k_train, batch_size=1)#, shuffle=True, num_workers=2)
# # flickr30k_testloader = torch.utils.data.DataLoader(flickr30k_test, batch_size=1)#, shuffle=False, num_workers=2)


# #-----------------------------------------------

mydataloader = [
    [DataLoader(caltech101_train, batch_size=1), DataLoader(caltech101_test, batch_size=1)],
    [DataLoader(cifar10_train, batch_size=1), DataLoader(cifar10_test, batch_size=1)],
    [DataLoader(oxford102flower_train, batch_size=1), DataLoader(oxford102flower_test, batch_size=1)],
    [DataLoader(coco_train, batch_size=1), DataLoader(coco_val, batch_size=1)],
    # [flickr30k_trainloader, flickr30k_testloader],
    # [caltech101_trainloader, caltech101_testloader],
    # [cifar_trainloader, cifar_testloader],
    # [DataLoader(oxford5k_train, batch_size=1), DataLoader(oxford5k_test, batch_size=1)],
    # [DataLoader(corel5k_train, batch_size=1), DataLoader(corel5k_test, batch_size=1)],
    # [DataLoader(inriaHoliday_train, batch_size=1), DataLoader(inriaHoliday_test, batch_size=1)]
]
# mydataloader = [
#     [None, None],
#     [None, None],
#     [None, None],
#     [DataLoader(coco_train, batch_size=1), DataLoader(coco_val, batch_size=1)],
#     # [flickr30k_trainloader, flickr30k_testloader],
#     # [caltech101_trainloader, caltech101_testloader],
#     # [cifar_trainloader, cifar_testloader],
#     # [DataLoader(oxford5k_train, batch_size=1), DataLoader(oxford5k_test, batch_size=1)],
#     # [DataLoader(corel5k_train, batch_size=1), DataLoader(corel5k_test, batch_size=1)],
#     # [DataLoader(inriaHoliday_train, batch_size=1), DataLoader(inriaHoliday_test, batch_size=1)]
# ]
BackBoneInstance = [
    # Resnet18_custom_best(),
    Resnet18Descriptor(),
    Resnet50Descriptor(), 
    MobileNetV3Feature(),
    MobileNetV3Feature_large(),
    SwinTransformer_default()
    # tinyvit(),
    # tinyvit_small(),
    # MyEfficientViT()
]
IndexingInstance = [
    #resnet18
    FaissRawIndex(512),#0---------
    
    #resnet50
    FaissRawIndex(2048), #1--------
    
    #mobilenetv3_small
    FaissRawIndex(576),#2----------
    
    #mobilenetv3_large
    FaissRawIndex(960),#3----------
    
    #SwinTransformer_default
    FaissRawIndex(768), #4----------
    
]

metadata_info = [
    [["caltech101", "best_resnet18_RawIndex"],
     ["caltech101", "resnet50_RawIndex"],
     ["caltech101", "MobileNetV3_small_custom_RawIndex"],
     ["caltech101", "MobileNetV3_large_RawIndex"],
     ["caltech101", "SwinTransformer_default_RawIndex"]],
    [["cifar10", "best_resnet18_RawIndex"],
     ["cifar10", "resnet50_RawIndex"],
     ["cifar10", "MobileNetV3_small_custom_RawIndex"],
     ["cifar10", "MobileNetV3_large_RawIndex"],
     ["cifar10", "SwinTransformer_default_RawIndex"]],
    [["oxford102flower", "best_resnet18_RawIndex"],
     ["oxford102flower", "resnet50_RawIndex"],
     ["oxford102flower", "MobileNetV3_small_custom_RawIndex"],
     ["oxford102flower", "MobileNetV3_large_RawIndex"],
     ["oxford102flower", "SwinTransformer_default_RawIndex"]],
    [["coco-2017", "best_resnet18_RawIndex"],
     ["coco-2017", "resnet50_RawIndex"],
     ["coco-2017", "MobileNetV3_small_custom_RawIndex"],
     ["coco-2017", "MobileNetV3_large_RawIndex"],
     ["coco-2017", "SwinTransformer_default_RawIndex"]],
    # [["INRIA_Holiday", "best_resnet18_faisslsh"],
    #  ["INRIA_Holiday", "MobileNetV3_small_custom_faisslsh"],
    #  ["INRIA_Holiday", "MobileNetV3_large_faisslsh"],
    #  ["INRIA_Holiday", "TinyViT_small_RawIndex"],
    #  ["INRIA_Holiday", "EfficientViT-M0_RawIndex"]]   
]

############################################################################################
################################# CBIR INSTANCE ############################################
############################################################################################
head_output = None
TestSearch = [
    [
        CBIR(BackBoneInstance[0], IndexingInstance[0], metadata=metadata_info[0][0]),
        CBIR(BackBoneInstance[1], IndexingInstance[1], metadata=metadata_info[0][1]),
        CBIR(BackBoneInstance[2], IndexingInstance[2], metadata=metadata_info[0][2]),
        CBIR(BackBoneInstance[3], IndexingInstance[3], metadata=metadata_info[0][3]),
        CBIR(BackBoneInstance[4], IndexingInstance[4], metadata=metadata_info[0][4])  
    ],
    [
        CBIR(BackBoneInstance[0], IndexingInstance[0], metadata=metadata_info[1][0]),
        CBIR(BackBoneInstance[1], IndexingInstance[1], metadata=metadata_info[1][1]),
        CBIR(BackBoneInstance[2], IndexingInstance[2], metadata=metadata_info[1][2]),
        CBIR(BackBoneInstance[3], IndexingInstance[3], metadata=metadata_info[1][3]),
        CBIR(BackBoneInstance[4], IndexingInstance[4], metadata=metadata_info[1][4])   
    ],
    [
        CBIR(BackBoneInstance[0], IndexingInstance[0], metadata=metadata_info[2][0]),
        CBIR(BackBoneInstance[1], IndexingInstance[1], metadata=metadata_info[2][1]),
        CBIR(BackBoneInstance[2], IndexingInstance[2], metadata=metadata_info[2][2]),
        CBIR(BackBoneInstance[3], IndexingInstance[3], metadata=metadata_info[2][3]), 
        CBIR(BackBoneInstance[4], IndexingInstance[4], metadata=metadata_info[2][4])  
    ],
    [
        CBIR(BackBoneInstance[0], IndexingInstance[0], metadata=metadata_info[3][0]),
        CBIR(BackBoneInstance[1], IndexingInstance[1], metadata=metadata_info[3][1]),
        CBIR(BackBoneInstance[2], IndexingInstance[2], metadata=metadata_info[3][2]),
        CBIR(BackBoneInstance[3], IndexingInstance[3], metadata=metadata_info[3][3]),
        CBIR(BackBoneInstance[4], IndexingInstance[4], metadata=metadata_info[3][4])  
    ],
    
]
###########################################################################################

folderpath_retrieve =  [[
                          'D:\\temp\\thesis_result - Copy\lsh_query\Caltech101\\resnet18',
                          'D:\\temp\\thesis_result - Copy\\lsh_query\\Caltech101\\resnet50', 
                          'D:\\temp\\thesis_result - Copy\\lsh_query\\Caltech101\\mobilenetv3_small',
                          'D:\\temp\\thesis_result - Copy\\lsh_query\\Caltech101\\mobilenetv3_large', 
                        ],
                        [
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Cifar10\\resnet18',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Cifar10\\resnet50',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Cifar10\\mobilenetv3_small',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Cifar10\\mobilenetv3_large',
                        ],
                        [
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Oxford102Flower\\resnet18',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Oxford102Flower\\resnet50',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Oxford102Flower\\mobilenetv3_small',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Oxford102Flower\\mobilenetv3_large',
                        ],
                        [
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Coco2017\\resnet18',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Coco2017\\resnet50',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Coco2017\\mobilenetv3_small',
                            'D:\\temp\\thesis_result - Copy\\lsh_query\\Coco2017\\mobilenetv3_large',
                        ]
                        ]

############################################################################################
################################# PERFORM INDEXING AND RETRIEVING ##########################
############################################################################################
#TESTING FOR RAWINDEX
perform_index = []
for i in range(0, 4, 1): perform_index.append([True]*5)
# for i in range(0, 4, 1): perform_index[i][4] = True
# for i in [1,3]: perform_index[i][4] = True
# perform_index[3][4] = True

perform_eval = []
for i in range(0, 4, 1): perform_eval.append([True]*5)
# for i in range(0, 4, 1): perform_eval[i][4] = True
# for i in [1,3]: perform_eval[i][4] = True
# perform_eval[3][4] = True

k_top = [5]

##############################################################################################
for idx_db in range(0, len(TestSearch), 1):
    print("================================= Database",idx_db + 1, "=================================")
    ii = 0
    for idx_cbir in range(0, len(TestSearch[idx_db]), 1):
        if perform_index[idx_db][idx_cbir] == True:
            TestSearch[idx_db][idx_cbir].indexDB(mydataloader[idx_db][0])
        # Evaluate phase
        if perform_eval[idx_db][idx_cbir] == True:
            for k in k_top:
                TestSearch[idx_db][idx_cbir].evalRetrieval(mydataloader[idx_db][1], k)
                # TestSearch[idx_db][idx_cbir].evalOnSingleQuery(folderpath_retrieve[idx_db][ii])
            ii = ii + 1
        print("-------------------------------------------------------------------------------")