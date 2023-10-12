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

# CALTECH-101 DATASET
PATH_CALTECH101 = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\caltech-101\\101_ObjectCategories"
caltech101ds = CaltechDataset(PATH_CALTECH101, transform_img)
caltech_train_indices, caltech_test_indices = train_test_split(range(len(caltech101ds)),stratify=caltech101ds.getLabels(), test_size=0.2)
caltech101_train = torch.utils.data.Subset(caltech101ds, caltech_train_indices)
caltech101_test = torch.utils.data.Subset(caltech101ds, caltech_test_indices)
# CIFAR-10 DATASET
PATH_CIFAR10 = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\cifar-10\\train"
csv_label_path = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\cifar-10\\trainLabels.csv"

cifar10ds = CifarDataset(PATH_CIFAR10, csv_label_path, transform_img)
cifar10_train_indices, cifar10_test_indices = train_test_split(range(len(cifar10ds)),stratify=cifar10ds.getLabels(), test_size=0.2)
cifar10_train = torch.utils.data.Subset(cifar10ds, cifar10_train_indices)
cifar10_test = torch.utils.data.Subset(cifar10ds, cifar10_test_indices)
# OXFORD-102-FLOWER DATASET
PATH_OXFORD102FLOWERS_TRAIN = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\102flowers_categorized\\dataset\\train"
PATH_OXFORD102FLOWERS_TEST = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\102flowers_categorized\\dataset\\valid"
oxford102flower_train = Oxford102Flower(PATH_OXFORD102FLOWERS_TRAIN, transform_img)
oxford102flower_test = Oxford102Flower(PATH_OXFORD102FLOWERS_TEST, transform_img)

# NUS-WIDE DATASET

# MS-COCO 2017 DATASET
dataDir = 'D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\coco2017'

dataType_val = 'val2017'
dataType_train = 'train2017'

annFile_train = f'{dataDir}/annotations/instances_{dataType_train}.json'
annFile_val = f'{dataDir}/annotations/instances_{dataType_val}.json'

coco_train = CustomCocoDataset(root=f'{dataDir}/{dataType_train}', annFile=annFile_train, transform=transform_img)
coco_val = CustomCocoDataset(root=f'{dataDir}/{dataType_val}', annFile=annFile_val, transform=transform_img)

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
BackBoneInstance = [
    # Resnet18_custom_best(),
    Resnet18Descriptor(),
    Resnet50Descriptor(), 
    MobileNetV3Feature(),
    MobileNetV3Feature_large(),
    SwinTransformer_default(),
    tinyvit(),
    # tinyvit_small(),
    # MyEfficientViT()
]

############################################################################################
################################# CBIR INSTANCE ############################################
############################################################################################
# INDEX INSTANCES
RawIndex_instance = [
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
    #TinyViT Transformer
    FaissRawIndex(360) #5----------    
]
feature_dim = [512, 2048, 576, 960, 768, 360]
FaissLSH_bitdepth = [16, 32, 64, 128, 256, 512, 1024, 2048]
CustomLSH_bitdepth = [1, 2, 3, 4, 5, 6, 7, 8]

FaissLSHIndex_instance = [
    [FaissLSHIndex(512, 16),
    FaissLSHIndex(512, 32),
    FaissLSHIndex(512, 64),
    FaissLSHIndex(512, 128),
    FaissLSHIndex(512, 256),
    FaissLSHIndex(512, 512),
    FaissLSHIndex(512, 1024),
    FaissLSHIndex(512, 2048)],
    [FaissLSHIndex(2048, 16),
    FaissLSHIndex(2048, 32),
    FaissLSHIndex(2048, 64),
    FaissLSHIndex(2048, 128),
    FaissLSHIndex(2048, 256),
    FaissLSHIndex(2048, 512),
    FaissLSHIndex(2048, 1024),
    FaissLSHIndex(2048, 2048)],
    [FaissLSHIndex(576, 16),
    FaissLSHIndex(576, 32),
    FaissLSHIndex(576, 64),
    FaissLSHIndex(576, 128),
    FaissLSHIndex(576, 256),
    FaissLSHIndex(576, 512),
    FaissLSHIndex(576, 1024),
    FaissLSHIndex(576, 2048)],
    [FaissLSHIndex(960, 16),
    FaissLSHIndex(960, 32),
    FaissLSHIndex(960, 64),
    FaissLSHIndex(960, 128),
    FaissLSHIndex(960, 256),
    FaissLSHIndex(960, 512),
    FaissLSHIndex(960, 1024),
    FaissLSHIndex(960, 2048)],
    [FaissLSHIndex(768, 16),
    FaissLSHIndex(768, 32),
    FaissLSHIndex(768, 64),
    FaissLSHIndex(768, 128),
    FaissLSHIndex(768, 256),
    FaissLSHIndex(768, 512),
    FaissLSHIndex(768, 1024),
    FaissLSHIndex(768, 2048)],
    [FaissLSHIndex(360, 16),
    FaissLSHIndex(360, 32),
    FaissLSHIndex(360, 64),
    FaissLSHIndex(360, 128),
    FaissLSHIndex(360, 256),
    FaissLSHIndex(360, 512),
    FaissLSHIndex(360, 1024),
    FaissLSHIndex(360, 2048)]

]

CustomLSHIndex_instance = [
    [CustomLSHIndex(1, 512),
    CustomLSHIndex(2, 512),
    CustomLSHIndex(3, 512),
    CustomLSHIndex(4, 512),
    CustomLSHIndex(5, 512),
    CustomLSHIndex(6, 512),
    CustomLSHIndex(7, 512),
    CustomLSHIndex(8, 512)],
    [CustomLSHIndex(1, 2048),
    CustomLSHIndex(2, 2048),
    CustomLSHIndex(3, 2048),
    CustomLSHIndex(4, 2048),
    CustomLSHIndex(5, 2048),
    CustomLSHIndex(6, 2048),
    CustomLSHIndex(7, 2048),
    CustomLSHIndex(8, 2048)],
    [CustomLSHIndex(1, 576),
    CustomLSHIndex(2, 576),
    CustomLSHIndex(3,576),
    CustomLSHIndex(4,576),
    CustomLSHIndex(5,576),
    CustomLSHIndex(6,576),
    CustomLSHIndex(7,576),
    CustomLSHIndex(8,576)],    
    [CustomLSHIndex(1,960),
    CustomLSHIndex(2,960),
    CustomLSHIndex(3,960),
    CustomLSHIndex(4,960),
    CustomLSHIndex(5,960),
    CustomLSHIndex(6,960),
    CustomLSHIndex(7,960),
    CustomLSHIndex(8,960)],    
    [CustomLSHIndex(1,768),
    CustomLSHIndex(2,768),
    CustomLSHIndex(3,768),
    CustomLSHIndex(4,768),
    CustomLSHIndex(5,768),
    CustomLSHIndex(6,768),
    CustomLSHIndex(7,768),
    CustomLSHIndex(8,768)],    
    [CustomLSHIndex(1,360),
    CustomLSHIndex(2,360),
    CustomLSHIndex(3,360),
    CustomLSHIndex(4,360),
    CustomLSHIndex(5,360),
    CustomLSHIndex(6,360),
    CustomLSHIndex(7,360),
    CustomLSHIndex(8,360)]
]

# METADATA
RawIndex_metadata = [
    [["caltech101", "resnet18_RawIndex"],
     ["caltech101", "resnet50_RawIndex"],
     ["caltech101", "MobileNetV3_small_RawIndex"],
     ["caltech101", "MobileNetV3_large_RawIndex"],
     ["caltech101", "SwinTransformer_default_RawIndex"],
     ["caltech101", "ViTransformer_default_RawIndex"]],
    [["cifar10", "resnet18_RawIndex"],
     ["cifar10", "resnet50_RawIndex"],
     ["cifar10", "MobileNetV3_small_RawIndex"],
     ["cifar10", "MobileNetV3_large_RawIndex"],
     ["cifar10", "SwinTransformer_default_RawIndex"],
     ["cifar10", "ViTransformer_default_RawIndex"]],
    [["oxford102flower", "resnet18_RawIndex"],
     ["oxford102flower", "resnet50_RawIndex"],
     ["oxford102flower", "MobileNetV3_small_RawIndex"],
     ["oxford102flower", "MobileNetV3_large_RawIndex"],
     ["oxford102flower", "SwinTransformer_default_RawIndex"],
     ["oxford102flower", "ViTransformer_default_RawIndex"]],
    [["coco-2017", "resnet18_RawIndex"],
     ["coco-2017", "resnet50_RawIndex"],
     ["coco-2017", "MobileNetV3_small_RawIndex"],
     ["coco-2017", "MobileNetV3_large_RawIndex"],
     ["coco-2017", "SwinTransformer_default_RawIndex"],
     ["coco-2017", "ViTransformer_default_RawIndex"]],
    # [["INRIA_Holiday", "resnet18_faisslsh"],
    #  ["INRIA_Holiday", "MobileNetV3_small_faisslsh"],
    #  ["INRIA_Holiday", "MobileNetV3_large_faisslsh"],
    #  ["INRIA_Holiday", "TinyViT_small_RawIndex"],
    #  ["INRIA_Holiday", "EfficientViT-M0_RawIndex"]]   
]
FaissLSH_metadata = [
    # caltech101
    [
    [["caltech101", "resnet18_FaissLSH_16_bits"],
    ["caltech101", "resnet18_FaissLSH_32_bits"],
    ["caltech101", "resnet18_FaissLSH_64_bits"],
    ["caltech101", "resnet18_FaissLSH_128_bits"],
    ["caltech101", "resnet18_FaissLSH_256_bits"],
    ["caltech101", "resnet18_FaissLSH_512_bits"],
    ["caltech101", "resnet18_FaissLSH_1024_bits"],
    ["caltech101", "resnet18_FaissLSH_2048_bits"]],
    [["caltech101", "resnet50_FaissLSH_16_bits"],
    ["caltech101", "resnet50_FaissLSH_32_bits"],
    ["caltech101", "resnet50_FaissLSH_64_bits"],
    ["caltech101", "resnet50_FaissLSH_128_bits"],
    ["caltech101", "resnet50_FaissLSH_256_bits"],
    ["caltech101", "resnet50_FaissLSH_512_bits"],
    ["caltech101", "resnet50_FaissLSH_1024_bits"],
    ["caltech101", "resnet50_FaissLSH_2048_bits"]],
    [["caltech101", "mobilenetv3_small_FaissLSH_16_bits"],
    ["caltech101", "mobilenetv3_small_FaissLSH_32_bits"],
    ["caltech101", "mobilenetv3_small_FaissLSH_64_bits"],
    ["caltech101", "mobilenetv3_small_FaissLSH_128_bits"],
    ["caltech101", "mobilenetv3_small_FaissLSH_256_bits"],
    ["caltech101", "mobilenetv3_small_FaissLSH_512_bits"],
    ["caltech101", "mobilenetv3_small_FaissLSH_1024_bits"],
    ["caltech101", "mobilenetv3_small_FaissLSH_2048_bits"]], 
    [["caltech101", "mobilenetv3_large_FaissLSH_16_bits"],
    ["caltech101", "mobilenetv3_large_FaissLSH_32_bits"],
    ["caltech101", "mobilenetv3_large_FaissLSH_64_bits"],
    ["caltech101", "mobilenetv3_large_FaissLSH_128_bits"],
    ["caltech101", "mobilenetv3_large_FaissLSH_256_bits"],
    ["caltech101", "mobilenetv3_large_FaissLSH_512_bits"],
    ["caltech101", "mobilenetv3_large_FaissLSH_1024_bits"],
    ["caltech101", "mobilenetv3_large_FaissLSH_2048_bits"]]

    ],
    # cifar10
    [
    [["cifar10", "resnet18_FaissLSH_16_bits"],
    ["cifar10", "resnet18_FaissLSH_32_bits"],
    ["cifar10", "resnet18_FaissLSH_64_bits"],
    ["cifar10", "resnet18_FaissLSH_128_bits"],
    ["cifar10", "resnet18_FaissLSH_256_bits"],
    ["cifar10", "resnet18_FaissLSH_512_bits"],
    ["cifar10", "resnet18_FaissLSH_1024_bits"],
    ["cifar10", "resnet18_FaissLSH_2048_bits"]],
    [["cifar10", "resnet50_FaissLSH_16_bits"],
    ["cifar10", "resnet50_FaissLSH_32_bits"],
    ["cifar10", "resnet50_FaissLSH_64_bits"],
    ["cifar10", "resnet50_FaissLSH_128_bits"],
    ["cifar10", "resnet50_FaissLSH_256_bits"],
    ["cifar10", "resnet50_FaissLSH_512_bits"],
    ["cifar10", "resnet50_FaissLSH_1024_bits"],
    ["cifar10", "resnet50_FaissLSH_2048_bits"]],
    [["cifar10", "mobilenetv3_small_FaissLSH_16_bits"],
    ["cifar10", "mobilenetv3_small_FaissLSH_32_bits"],
    ["cifar10", "mobilenetv3_small_FaissLSH_64_bits"],
    ["cifar10", "mobilenetv3_small_FaissLSH_128_bits"],
    ["cifar10", "mobilenetv3_small_FaissLSH_256_bits"],
    ["cifar10", "mobilenetv3_small_FaissLSH_512_bits"],
    ["cifar10", "mobilenetv3_small_FaissLSH_1024_bits"],
    ["cifar10", "mobilenetv3_small_FaissLSH_2048_bits"]],
    [["cifar10", "mobilenetv3_large_FaissLSH_16_bits"],
    ["cifar10", "mobilenetv3_large_FaissLSH_32_bits"],
    ["cifar10", "mobilenetv3_large_FaissLSH_64_bits"],
    ["cifar10", "mobilenetv3_large_FaissLSH_128_bits"],
    ["cifar10", "mobilenetv3_large_FaissLSH_256_bits"],
    ["cifar10", "mobilenetv3_large_FaissLSH_512_bits"],
    ["cifar10", "mobilenetv3_large_FaissLSH_1024_bits"],
    ["cifar10", "mobilenetv3_large_FaissLSH_2048_bits"]]

    ],
    # 102flowers
    [
    [["102flower", "resnet18_FaissLSH_16_bits"],
    ["102flower", "resnet18_FaissLSH_32_bits"],
    ["102flower", "resnet18_FaissLSH_64_bits"],
    ["102flower", "resnet18_FaissLSH_128_bits"],
    ["102flower", "resnet18_FaissLSH_256_bits"],
    ["102flower", "resnet18_FaissLSH_512_bits"],
    ["102flower", "resnet18_FaissLSH_1024_bits"],
    ["102flower", "resnet18_FaissLSH_2048_bits"]],
    [["102flower", "resnet50_FaissLSH_16_bits"],
    ["102flower", "resnet50_FaissLSH_32_bits"],
    ["102flower", "resnet50_FaissLSH_64_bits"],
    ["102flower", "resnet50_FaissLSH_128_bits"],
    ["102flower", "resnet50_FaissLSH_256_bits"],
    ["102flower", "resnet50_FaissLSH_512_bits"],
    ["102flower", "resnet50_FaissLSH_1024_bits"],
    ["102flower", "resnet50_FaissLSH_2048_bits"]],
    [["102flower", "mobilenetv3_small_FaissLSH_16_bits"],
    ["102flower", "mobilenetv3_small_FaissLSH_32_bits"],
    ["102flower", "mobilenetv3_small_FaissLSH_64_bits"],
    ["102flower", "mobilenetv3_small_FaissLSH_128_bits"],
    ["102flower", "mobilenetv3_small_FaissLSH_256_bits"],
    ["102flower", "mobilenetv3_small_FaissLSH_512_bits"],
    ["102flower", "mobilenetv3_small_FaissLSH_1024_bits"],
    ["102flower", "mobilenetv3_small_FaissLSH_2048_bits"]],
    [["102flower", "mobilenetv3_large_FaissLSH_16_bits"],
    ["102flower", "mobilenetv3_large_FaissLSH_32_bits"],
    ["102flower", "mobilenetv3_large_FaissLSH_64_bits"],
    ["102flower", "mobilenetv3_large_FaissLSH_128_bits"],
    ["102flower", "mobilenetv3_large_FaissLSH_256_bits"],
    ["102flower", "mobilenetv3_large_FaissLSH_512_bits"],
    ["102flower", "mobilenetv3_large_FaissLSH_1024_bits"],
    ["102flower", "mobilenetv3_large_FaissLSH_2048_bits"]]
    ],
    # Coco2017
    [
   [["coco2017", "resnet18_FaissLSH_16_bits"],
    ["coco2017", "resnet18_FaissLSH_32_bits"],
    ["coco2017", "resnet18_FaissLSH_64_bits"],
    ["coco2017", "resnet18_FaissLSH_128_bits"],
    ["coco2017", "resnet18_FaissLSH_256_bits"],
    ["coco2017", "resnet18_FaissLSH_512_bits"],
    ["coco2017", "resnet18_FaissLSH_1024_bits"],
    ["coco2017", "resnet18_FaissLSH_2048_bits"]],
    [["coco2017", "resnet50_FaissLSH_16_bits"],
    ["coco2017", "resnet50_FaissLSH_32_bits"],
    ["coco2017", "resnet50_FaissLSH_64_bits"],
    ["coco2017", "resnet50_FaissLSH_128_bits"],
    ["coco2017", "resnet50_FaissLSH_256_bits"],
    ["coco2017", "resnet50_FaissLSH_512_bits"],
    ["coco2017", "resnet50_FaissLSH_1024_bits"],
    ["coco2017", "resnet50_FaissLSH_2048_bits"]],
    [["coco2017", "mobilenetv3_small_FaissLSH_16_bits"],
    ["coco2017", "mobilenetv3_small_FaissLSH_32_bits"],
    ["coco2017", "mobilenetv3_small_FaissLSH_64_bits"],
    ["coco2017", "mobilenetv3_small_FaissLSH_128_bits"],
    ["coco2017", "mobilenetv3_small_FaissLSH_256_bits"],
    ["coco2017", "mobilenetv3_small_FaissLSH_512_bits"],
    ["coco2017", "mobilenetv3_small_FaissLSH_1024_bits"],
    ["coco2017", "mobilenetv3_small_FaissLSH_2048_bits"]],
    [["coco2017", "mobilenetv3_large_FaissLSH_16_bits"],
    ["coco2017", "mobilenetv3_large_FaissLSH_32_bits"],
    ["coco2017", "mobilenetv3_large_FaissLSH_64_bits"],
    ["coco2017", "mobilenetv3_large_FaissLSH_128_bits"],
    ["coco2017", "mobilenetv3_large_FaissLSH_256_bits"],
    ["coco2017", "mobilenetv3_large_FaissLSH_512_bits"],
    ["coco2017", "mobilenetv3_large_FaissLSH_1024_bits"],
    ["coco2017", "mobilenetv3_large_FaissLSH_2048_bits"]]
    ]

]
CustomLSH_metadata = [
    # caltech101
    [
    [["caltech101", "resnet18_CustomLSH_1_bits"],
    ["caltech101", "resnet18_CustomLSH_2_bits"],
    ["caltech101", "resnet18_CustomLSH_3_bits"],
    ["caltech101", "resnet18_CustomLSH_4_bits"],
    ["caltech101", "resnet18_CustomLSH_5_bits"],
    ["caltech101", "resnet18_CustomLSH_6_bits"],
    ["caltech101", "resnet18_CustomLSH_7_bits"],
    ["caltech101", "resnet18_CustomLSH_8_bits"]],
    [["caltech101", "resnet50_CustomLSH_1_bits"],
    ["caltech101", "resnet50_CustomLSH_2_bits"],
    ["caltech101", "resnet50_CustomLSH_3_bits"],
    ["caltech101", "resnet50_CustomLSH_4_bits"],
    ["caltech101", "resnet50_CustomLSH_5_bits"],
    ["caltech101", "resnet50_CustomLSH_6_bits"],
    ["caltech101", "resnet50_CustomLSH_7_bits"],
    ["caltech101", "resnet50_CustomLSH_8_bits"]],
    [["caltech101", "mobilenetv3_small_CustomLSH_1_bits"],
    ["caltech101", "mobilenetv3_small_CustomLSH_2_bits"],
    ["caltech101", "mobilenetv3_small_CustomLSH_3_bits"],
    ["caltech101", "mobilenetv3_small_CustomLSH_4_bits"],
    ["caltech101", "mobilenetv3_small_CustomLSH_5_bits"],
    ["caltech101", "mobilenetv3_small_CustomLSH_6_bits"],
    ["caltech101", "mobilenetv3_small_CustomLSH_7_bits"],
    ["caltech101", "mobilenetv3_small_CustomLSH_8_bits"]],
    [["caltech101", "mobilenetv3_large_CustomLSH_1_bits"],
    ["caltech101", "mobilenetv3_large_CustomLSH_2_bits"],
    ["caltech101", "mobilenetv3_large_CustomLSH_3_bits"],
    ["caltech101", "mobilenetv3_large_CustomLSH_4_bits"],
    ["caltech101", "mobilenetv3_large_CustomLSH_5_bits"],
    ["caltech101", "mobilenetv3_large_CustomLSH_6_bits"],
    ["caltech101", "mobilenetv3_large_CustomLSH_7_bits"],
    ["caltech101", "mobilenetv3_large_CustomLSH_8_bits"]]

    ],
    # cifar10
    [
    [["cifar10", "resnet18_CustomLSH_1_bits"],
    ["cifar10", "resnet18_CustomLSH_2_bits"],
    ["cifar10", "resnet18_CustomLSH_3_bits"],
    ["cifar10", "resnet18_CustomLSH_4_bits"],
    ["cifar10", "resnet18_CustomLSH_5_bits"],
    ["cifar10", "resnet18_CustomLSH_6_bits"],
    ["cifar10", "resnet18_CustomLSH_7_bits"],
    ["cifar10", "resnet18_CustomLSH_8_bits"]],
    [["cifar10", "resnet50_CustomLSH_1_bits"],
    ["cifar10", "resnet50_CustomLSH_2_bits"],
    ["cifar10", "resnet50_CustomLSH_3_bits"],
    ["cifar10", "resnet50_CustomLSH_4_bits"],
    ["cifar10", "resnet50_CustomLSH_5_bits"],
    ["cifar10", "resnet50_CustomLSH_6_bits"],
    ["cifar10", "resnet50_CustomLSH_7_bits"],
    ["cifar10", "resnet50_CustomLSH_8_bits"]],
    [["cifar10", "mobilenetv3_small_CustomLSH_1_bits"],
    ["cifar10", "mobilenetv3_small_CustomLSH_2_bits"],
    ["cifar10", "mobilenetv3_small_CustomLSH_3_bits"],
    ["cifar10", "mobilenetv3_small_CustomLSH_4_bits"],
    ["cifar10", "mobilenetv3_small_CustomLSH_5_bits"],
    ["cifar10", "mobilenetv3_small_CustomLSH_6_bits"],
    ["cifar10", "mobilenetv3_small_CustomLSH_7_bits"],
    ["cifar10", "mobilenetv3_small_CustomLSH_8_bits"]],
    [["cifar10", "mobilenetv3_large_CustomLSH_1_bits"],
    ["cifar10", "mobilenetv3_large_CustomLSH_2_bits"],
    ["cifar10", "mobilenetv3_large_CustomLSH_3_bits"],
    ["cifar10", "mobilenetv3_large_CustomLSH_4_bits"],
    ["cifar10", "mobilenetv3_large_CustomLSH_5_bits"],
    ["cifar10", "mobilenetv3_large_CustomLSH_6_bits"],
    ["cifar10", "mobilenetv3_large_CustomLSH_7_bits"],
    ["cifar10", "mobilenetv3_large_CustomLSH_8_bits"]]

    ],
    # 102flowers
    [
    [["102flower", "resnet18_CustomLSH_1_bits"],
    ["102flower", "resnet18_CustomLSH_2_bits"],
    ["102flower", "resnet18_CustomLSH_3_bits"],
    ["102flower", "resnet18_CustomLSH_4_bits"],
    ["102flower", "resnet18_CustomLSH_5_bits"],
    ["102flower", "resnet18_CustomLSH_6_bits"],
    ["102flower", "resnet18_CustomLSH_7_bits"],
    ["102flower", "resnet18_CustomLSH_8_bits"]],
    [["102flower", "resnet50_CustomLSH_1_bits"],
    ["102flower", "resnet50_CustomLSH_2_bits"],
    ["102flower", "resnet50_CustomLSH_3_bits"],
    ["102flower", "resnet50_CustomLSH_4_bits"],
    ["102flower", "resnet50_CustomLSH_5_bits"],
    ["102flower", "resnet50_CustomLSH_6_bits"],
    ["102flower", "resnet50_CustomLSH_7_bits"],
    ["102flower", "resnet50_CustomLSH_8_bits"]],
    [["102flower", "mobilenetv3_small_CustomLSH_1_bits"],
    ["102flower", "mobilenetv3_small_CustomLSH_2_bits"],
    ["102flower", "mobilenetv3_small_CustomLSH_3_bits"],
    ["102flower", "mobilenetv3_small_CustomLSH_4_bits"],
    ["102flower", "mobilenetv3_small_CustomLSH_5_bits"],
    ["102flower", "mobilenetv3_small_CustomLSH_6_bits"],
    ["102flower", "mobilenetv3_small_CustomLSH_7_bits"],
    ["102flower", "mobilenetv3_small_CustomLSH_8_bits"]],
    [["102flower", "mobilenetv3_large_CustomLSH_1_bits"],
    ["102flower", "mobilenetv3_large_CustomLSH_2_bits"],
    ["102flower", "mobilenetv3_large_CustomLSH_3_bits"],
    ["102flower", "mobilenetv3_large_CustomLSH_4_bits"],
    ["102flower", "mobilenetv3_large_CustomLSH_5_bits"],
    ["102flower", "mobilenetv3_large_CustomLSH_6_bits"],
    ["102flower", "mobilenetv3_large_CustomLSH_7_bits"],
    ["102flower", "mobilenetv3_large_CustomLSH_8_bits"]]
    ],
    # Coco2017
    [
   [["coco2017", "resnet18_CustomLSH_1_bits"],
    ["coco2017", "resnet18_CustomLSH_2_bits"],
    ["coco2017", "resnet18_CustomLSH_3_bits"],
    ["coco2017", "resnet18_CustomLSH_4_bits"],
    ["coco2017", "resnet18_CustomLSH_5_bits"],
    ["coco2017", "resnet18_CustomLSH_6_bits"],
    ["coco2017", "resnet18_CustomLSH_7_bits"],
    ["coco2017", "resnet18_CustomLSH_8_bits"]],
    [["coco2017", "resnet50_CustomLSH_1_bits"],
    ["coco2017", "resnet50_CustomLSH_2_bits"],
    ["coco2017", "resnet50_CustomLSH_3_bits"],
    ["coco2017", "resnet50_CustomLSH_4_bits"],
    ["coco2017", "resnet50_CustomLSH_5_bits"],
    ["coco2017", "resnet50_CustomLSH_6_bits"],
    ["coco2017", "resnet50_CustomLSH_7_bits"],
    ["coco2017", "resnet50_CustomLSH_8_bits"]],
    [["coco2017", "mobilenetv3_small_CustomLSH_1_bits"],
    ["coco2017", "mobilenetv3_small_CustomLSH_2_bits"],
    ["coco2017", "mobilenetv3_small_CustomLSH_3_bits"],
    ["coco2017", "mobilenetv3_small_CustomLSH_4_bits"],
    ["coco2017", "mobilenetv3_small_CustomLSH_5_bits"],
    ["coco2017", "mobilenetv3_small_CustomLSH_6_bits"],
    ["coco2017", "mobilenetv3_small_CustomLSH_7_bits"],
    ["coco2017", "mobilenetv3_small_CustomLSH_8_bits"]],
    [["coco2017", "mobilenetv3_large_CustomLSH_1_bits"],
    ["coco2017", "mobilenetv3_large_CustomLSH_2_bits"],
    ["coco2017", "mobilenetv3_large_CustomLSH_3_bits"],
    ["coco2017", "mobilenetv3_large_CustomLSH_4_bits"],
    ["coco2017", "mobilenetv3_large_CustomLSH_5_bits"],
    ["coco2017", "mobilenetv3_large_CustomLSH_6_bits"],
    ["coco2017", "mobilenetv3_large_CustomLSH_7_bits"],
    ["coco2017", "mobilenetv3_large_CustomLSH_8_bits"]]
    ]

]
Index_instances = [
    RawIndex_instance,
    FaissLSHIndex_instance,
    CustomLSHIndex_instance
]
metadata = [
    RawIndex_metadata,
    FaissLSH_metadata,
    CustomLSH_metadata
]

# CBIR INSTANCES
# RawIndex: 0
# FaissLSH index: 1
# CustomLSH index: 2
lshindex_idx = 1
rawindex_idx = 0
LSHTestSearch = [
    [
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][0], metadata=metadata[lshindex_idx][0][0][0]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][1], metadata=metadata[lshindex_idx][0][0][1]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][2], metadata=metadata[lshindex_idx][0][0][2]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][3], metadata=metadata[lshindex_idx][0][0][3]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][4], metadata=metadata[lshindex_idx][0][0][4]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][5], metadata=metadata[lshindex_idx][0][0][5]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][6], metadata=metadata[lshindex_idx][0][0][6]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][7], metadata=metadata[lshindex_idx][0][0][7]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][0], metadata=metadata[lshindex_idx][0][1][0]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][1], metadata=metadata[lshindex_idx][0][1][1]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][2], metadata=metadata[lshindex_idx][0][1][2]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][3], metadata=metadata[lshindex_idx][0][1][3]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][4], metadata=metadata[lshindex_idx][0][1][4]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][5], metadata=metadata[lshindex_idx][0][1][5]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][6], metadata=metadata[lshindex_idx][0][1][6]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][7], metadata=metadata[lshindex_idx][0][1][7]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][0], metadata=metadata[lshindex_idx][0][2][0]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][1], metadata=metadata[lshindex_idx][0][2][1]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][2], metadata=metadata[lshindex_idx][0][2][2]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][3], metadata=metadata[lshindex_idx][0][2][3]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][4], metadata=metadata[lshindex_idx][0][2][4]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][5], metadata=metadata[lshindex_idx][0][2][5]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][6], metadata=metadata[lshindex_idx][0][2][6]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][7], metadata=metadata[lshindex_idx][0][2][7]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][0], metadata=metadata[lshindex_idx][0][3][0]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][1], metadata=metadata[lshindex_idx][0][3][1]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][2], metadata=metadata[lshindex_idx][0][3][2]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][3], metadata=metadata[lshindex_idx][0][3][3]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][4], metadata=metadata[lshindex_idx][0][3][4]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][5], metadata=metadata[lshindex_idx][0][3][5]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][6], metadata=metadata[lshindex_idx][0][3][6]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][7], metadata=metadata[lshindex_idx][0][3][7])
        
    ],
    [
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][0], metadata=metadata[lshindex_idx][1][0][0]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][1], metadata=metadata[lshindex_idx][1][0][1]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][2], metadata=metadata[lshindex_idx][1][0][2]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][3], metadata=metadata[lshindex_idx][1][0][3]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][4], metadata=metadata[lshindex_idx][1][0][4]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][5], metadata=metadata[lshindex_idx][1][0][5]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][6], metadata=metadata[lshindex_idx][1][0][6]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][7], metadata=metadata[lshindex_idx][1][0][7]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][0], metadata=metadata[lshindex_idx][1][1][0]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][1], metadata=metadata[lshindex_idx][1][1][1]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][2], metadata=metadata[lshindex_idx][1][1][2]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][3], metadata=metadata[lshindex_idx][1][1][3]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][4], metadata=metadata[lshindex_idx][1][1][4]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][5], metadata=metadata[lshindex_idx][1][1][5]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][6], metadata=metadata[lshindex_idx][1][1][6]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][7], metadata=metadata[lshindex_idx][1][1][7]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][0], metadata=metadata[lshindex_idx][1][2][0]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][1], metadata=metadata[lshindex_idx][1][2][1]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][2], metadata=metadata[lshindex_idx][1][2][2]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][3], metadata=metadata[lshindex_idx][1][2][3]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][4], metadata=metadata[lshindex_idx][1][2][4]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][5], metadata=metadata[lshindex_idx][1][2][5]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][6], metadata=metadata[lshindex_idx][1][2][6]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][7], metadata=metadata[lshindex_idx][1][2][7]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][0], metadata=metadata[lshindex_idx][1][3][0]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][1], metadata=metadata[lshindex_idx][1][3][1]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][2], metadata=metadata[lshindex_idx][1][3][2]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][3], metadata=metadata[lshindex_idx][1][3][3]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][4], metadata=metadata[lshindex_idx][1][3][4]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][5], metadata=metadata[lshindex_idx][1][3][5]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][6], metadata=metadata[lshindex_idx][1][3][6]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][7], metadata=metadata[lshindex_idx][1][3][7])
        
    ],
    [
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][0], metadata=metadata[lshindex_idx][2][0][0]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][1], metadata=metadata[lshindex_idx][2][0][1]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][2], metadata=metadata[lshindex_idx][2][0][2]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][3], metadata=metadata[lshindex_idx][2][0][3]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][4], metadata=metadata[lshindex_idx][2][0][4]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][5], metadata=metadata[lshindex_idx][2][0][5]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][6], metadata=metadata[lshindex_idx][2][0][6]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][7], metadata=metadata[lshindex_idx][2][0][7]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][0], metadata=metadata[lshindex_idx][2][1][0]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][1], metadata=metadata[lshindex_idx][2][1][1]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][2], metadata=metadata[lshindex_idx][2][1][2]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][3], metadata=metadata[lshindex_idx][2][1][3]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][4], metadata=metadata[lshindex_idx][2][1][4]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][5], metadata=metadata[lshindex_idx][2][1][5]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][6], metadata=metadata[lshindex_idx][2][1][6]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][7], metadata=metadata[lshindex_idx][2][1][7]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][0], metadata=metadata[lshindex_idx][2][2][0]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][1], metadata=metadata[lshindex_idx][2][2][1]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][2], metadata=metadata[lshindex_idx][2][2][2]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][3], metadata=metadata[lshindex_idx][2][2][3]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][4], metadata=metadata[lshindex_idx][2][2][4]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][5], metadata=metadata[lshindex_idx][2][2][5]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][6], metadata=metadata[lshindex_idx][2][2][6]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][7], metadata=metadata[lshindex_idx][2][2][7]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][0], metadata=metadata[lshindex_idx][2][3][0]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][1], metadata=metadata[lshindex_idx][2][3][1]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][2], metadata=metadata[lshindex_idx][2][3][2]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][3], metadata=metadata[lshindex_idx][2][3][3]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][4], metadata=metadata[lshindex_idx][2][3][4]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][5], metadata=metadata[lshindex_idx][2][3][5]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][6], metadata=metadata[lshindex_idx][2][3][6]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][7], metadata=metadata[lshindex_idx][2][3][7])
        
    ],
    [
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][0], metadata=metadata[lshindex_idx][3][0][0]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][1], metadata=metadata[lshindex_idx][3][0][1]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][2], metadata=metadata[lshindex_idx][3][0][2]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][3], metadata=metadata[lshindex_idx][3][0][3]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][4], metadata=metadata[lshindex_idx][3][0][4]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][5], metadata=metadata[lshindex_idx][3][0][5]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][6], metadata=metadata[lshindex_idx][3][0][6]),
        CBIR(BackBoneInstance[0], Index_instances[lshindex_idx][0][7], metadata=metadata[lshindex_idx][3][0][7]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][0], metadata=metadata[lshindex_idx][3][1][0]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][1], metadata=metadata[lshindex_idx][3][1][1]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][2], metadata=metadata[lshindex_idx][3][1][2]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][3], metadata=metadata[lshindex_idx][3][1][3]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][4], metadata=metadata[lshindex_idx][3][1][4]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][5], metadata=metadata[lshindex_idx][3][1][5]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][6], metadata=metadata[lshindex_idx][3][1][6]),
        CBIR(BackBoneInstance[1], Index_instances[lshindex_idx][1][7], metadata=metadata[lshindex_idx][3][1][7]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][0], metadata=metadata[lshindex_idx][3][2][0]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][1], metadata=metadata[lshindex_idx][3][2][1]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][2], metadata=metadata[lshindex_idx][3][2][2]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][3], metadata=metadata[lshindex_idx][3][2][3]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][4], metadata=metadata[lshindex_idx][3][2][4]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][5], metadata=metadata[lshindex_idx][3][2][5]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][6], metadata=metadata[lshindex_idx][3][2][6]),
        CBIR(BackBoneInstance[2], Index_instances[lshindex_idx][2][7], metadata=metadata[lshindex_idx][3][2][7]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][0], metadata=metadata[lshindex_idx][3][3][0]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][1], metadata=metadata[lshindex_idx][3][3][1]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][2], metadata=metadata[lshindex_idx][3][3][2]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][3], metadata=metadata[lshindex_idx][3][3][3]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][4], metadata=metadata[lshindex_idx][3][3][4]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][5], metadata=metadata[lshindex_idx][3][3][5]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][6], metadata=metadata[lshindex_idx][3][3][6]),
        CBIR(BackBoneInstance[3], Index_instances[lshindex_idx][3][7], metadata=metadata[lshindex_idx][3][3][7])
        
    ],

]
RawIndexTestSearch = [
    [
        CBIR(BackBoneInstance[0], Index_instances[rawindex_idx][0], metadata=metadata[0][0]),
        CBIR(BackBoneInstance[1], Index_instances[rawindex_idx][1], metadata=metadata[0][1]),
        CBIR(BackBoneInstance[2], Index_instances[rawindex_idx][2], metadata=metadata[0][2]),
        CBIR(BackBoneInstance[3], Index_instances[rawindex_idx][3], metadata=metadata[0][3]),
        CBIR(BackBoneInstance[4], Index_instances[rawindex_idx][4], metadata=metadata[0][4])  
    ],
    [
        CBIR(BackBoneInstance[0], Index_instances[rawindex_idx][0], metadata=metadata[1][0]),
        CBIR(BackBoneInstance[1], Index_instances[rawindex_idx][1], metadata=metadata[1][1]),
        CBIR(BackBoneInstance[2], Index_instances[rawindex_idx][2], metadata=metadata[1][2]),
        CBIR(BackBoneInstance[3], Index_instances[rawindex_idx][3], metadata=metadata[1][3]),
        CBIR(BackBoneInstance[4], Index_instances[rawindex_idx][4], metadata=metadata[1][4])   
    ],
    [
        CBIR(BackBoneInstance[0], Index_instances[rawindex_idx][0], metadata=metadata[2][0]),
        CBIR(BackBoneInstance[1], Index_instances[rawindex_idx][1], metadata=metadata[2][1]),
        CBIR(BackBoneInstance[2], Index_instances[rawindex_idx][2], metadata=metadata[2][2]),
        CBIR(BackBoneInstance[3], Index_instances[rawindex_idx][3], metadata=metadata[2][3]), 
        CBIR(BackBoneInstance[4], Index_instances[rawindex_idx][4], metadata=metadata[2][4])  
    ],
    [
        CBIR(BackBoneInstance[0], Index_instances[rawindex_idx][0], metadata=metadata[3][0]),
        CBIR(BackBoneInstance[1], Index_instances[rawindex_idx][1], metadata=metadata[3][1]),
        CBIR(BackBoneInstance[2], Index_instances[rawindex_idx][2], metadata=metadata[3][2]),
        CBIR(BackBoneInstance[3], Index_instances[rawindex_idx][3], metadata=metadata[3][3]),
        CBIR(BackBoneInstance[4], Index_instances[rawindex_idx][4], metadata=metadata[3][4])  
    ],
    
]
TestSearch = [
    RawIndexTestSearch,
    LSHTestSearch
]

############################################################################################
################################# PERFORM INDEXING AND RETRIEVING ##########################
############################################################################################


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
def main(search_idx):


    # TESTING FOR LSHINDEX AT BITDEPTH
    perform_index = []
    for i in range(0,4,1): perform_index.append([True]*len(TestSearch[search_idx][i]))
    if lshindex_idx == 2 and search_idx == 1: # LSH index with bit 1,2,3,4,5,6,7,8
        for db_i in range(0,4,1):
            for f_creator in range(0, len(BackBoneInstance), 1):
                for idx_i in [6, 7]:
                    cbir_idx = len(Index_instances[lshindex_idx][f_creator]) * len(f_creator) + idx_i
                    perform_index[db_i][cbir_idx] = False
    # for i in range(0,16,1): perform_index[0][i] = False
    # for i in range(7,32,1): perform_index[3][i] = True
    # for i in range(1,3,1): 
    #     for j in range(0,4,1):
    #         for k in range(0, 6, 1):
    #             cbir_idx = 8*j + k
    #             perform_index[i][cbir_idx] = True

    # for i in range(3,4,1):
    #     for j in [6, 14, 22, 30]: perform_index[i][j] = True
    # perform_index[3][22] = True
    # perform_index[0].append(True)
    perform_eval = []
    for i in range(0,4,1): perform_eval.append([True]*len(TestSearch[search_idx][i]))
    # for i in range(1,3,1): 
    #     for j in range(0,4,1):
    #         for k in range(0, 6, 1):
    #             cbir_idx = 8*j + k
    #             perform_eval[i][cbir_idx] = True
    # perform_eval[3][22] = True
    # perform_eval[3][30] = True
    # for i in range(3,4,1):
    #     for j in [6, 14, 22, 30]: perform_eval[i][j] = True
    # for i in range(0,16,1): perform_eval[0][i] = False
    # for i in range(6,32,1): perform_eval[3][i] = True
    # for j in range(1,4,1):
    #     for i in range(0,32,1): perform_eval[j][i] = False
    # perform_eval[3][6] = True
    # perform_eval[0].append(True)
    k_top = [5]

    for idx_db in range(0, len(TestSearch[search_idx]), 1):
        print("================================= Database",idx_db + 1, "=================================")
        ii = 0
        for idx_cbir in range(0, len(TestSearch[search_idx][idx_db]), 1):
            if perform_index[idx_db][idx_cbir] == True:
                TestSearch[search_idx][idx_db][idx_cbir].indexDB(mydataloader[idx_db][0])
            # Evaluate phase
            if perform_eval[idx_db][idx_cbir] == True:
                for k in k_top:
                    TestSearch[search_idx][idx_db][idx_cbir].evalRetrieval(mydataloader[idx_db][1], k)
                    # TestSearch[search_idx][idx_db][idx_cbir].evalOnSingleQuery(folderpath_retrieve[idx_db][ii])
                ii = ii + 1
            print("-------------------------------------------------------------------------------")

if __name__ == "__main__":
    main(sys.argv[1])