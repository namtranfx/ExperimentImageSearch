from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import torchvision

from source.data_handler import MyTransform, InriaHolidayDataset, CorelDataset, CaltechDataset, OxfordDataset, CifarDataset
from source.CBIR import CBIR
from source.features import *
from source.index import *
from source.ultis import to_rgb

############################################################################################
################################# DATASET PATH #############################################
############################################################################################

PATH_COREL5K = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\Corel-5k\\images"
PATH_COREL10K = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\Corel-10k"
PATH_HOLIDAY = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\The INRIA Holidays dataset\\jpg"
PATH_OXFORD5K = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\oxbuild_images"
PATH_CALTECH101 = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\caltech-101\\101_ObjectCategories"
PATH_CALTECH256 = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\caltech-101\\256_ObjectCategories"
PATH_CIFAR10 = "D:\\hcmus\\1. KHOA_LUAN\\current_work\\program_test\\dataset\\cifar-10\\train"

datasetpath = [PATH_COREL5K, PATH_COREL10K, PATH_HOLIDAY, PATH_OXFORD5K, PATH_CALTECH101, PATH_CALTECH256]

############################################################################################
################################# CUSTOM DATASET ###########################################
############################################################################################

caltech101ds = CaltechDataset(PATH_CALTECH101, MyTransform())
oxford5kds = OxfordDataset(PATH_OXFORD5K, MyTransform())
corel5kds = CorelDataset(PATH_COREL5K, MyTransform())
cifar10ds = CifarDataset(PATH_CIFAR10, MyTransform())
inriaHolidayds = InriaHolidayDataset(PATH_HOLIDAY, MyTransform())

# Split dataset into database(trainset) and evaluating set(testset)

caltech_train_indices, caltech_test_indices = train_test_split(range(len(caltech101ds)),stratify=caltech101ds.getLabels(), test_size=0.2)
oxford_train_indices, oxford_test_indices = train_test_split(range(len(oxford5kds)),stratify=oxford5kds.getLabels(), test_size=0.2)
corel5k_train_indices, corel5k_test_indices = train_test_split(range(len(corel5kds)),stratify=corel5kds.getLabels(), test_size=0.2)
cifar10_train_indices, cifar10_test_indices = train_test_split(range(len(cifar10ds)),stratify=cifar10ds.getLabels(), test_size=0.2)
inriaHoliday_train_indices, inriaHoliday_test_indices = train_test_split(range(len(inriaHolidayds)), shuffle=True, test_size=0.2)


caltech101_train = torch.utils.data.Subset(caltech101ds, caltech_train_indices)
caltech101_test = torch.utils.data.Subset(caltech101ds, caltech_test_indices)

oxford5k_train = torch.utils.data.Subset(oxford5kds, oxford_train_indices)
oxford5k_test = torch.utils.data.Subset(oxford5kds, oxford_test_indices)

corel5k_train = torch.utils.data.Subset(corel5kds, corel5k_train_indices)
corel5k_test = torch.utils.data.Subset(corel5kds, corel5k_test_indices)

cifar10_train = torch.utils.data.Subset(cifar10ds, cifar10_train_indices)
cifar10_test = torch.utils.data.Subset(cifar10ds, cifar10_test_indices)

inriaHoliday_train = torch.utils.data.Subset(inriaHolidayds, inriaHoliday_train_indices)
inriaHoliday_test = torch.utils.data.Subset(inriaHolidayds, inriaHoliday_test_indices)

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
    # [flickr30k_trainloader, flickr30k_testloader],
    # [caltech101_trainloader, caltech101_testloader],
    [DataLoader(cifar10_train, batch_size=1), DataLoader(cifar10_test, batch_size=1)],
    # [cifar_trainloader, cifar_testloader],
    [DataLoader(oxford5k_train, batch_size=1), DataLoader(oxford5k_test, batch_size=1)],
    [DataLoader(corel5k_train, batch_size=1), DataLoader(corel5k_test, batch_size=1)],
    
    [DataLoader(inriaHoliday_train, batch_size=1), DataLoader(inriaHoliday_test, batch_size=1)]
]
BackBoneInstance = [
    Resnet18_custom_best(),
    MobileNetV3Feature(),
    MobileNetV3Feature_large(),
    tinyvit(),
    tinyvit_small(),
    MyEfficientViT()
]
IndexingInstance = [
    FaissRawIndex(512),
    FaissLSHIndex(512, 128),
    FaissLSHIndex(576, 128),
    FaissLSHIndex(960, 128),
    AnnoyLSHIndex(512),
    AnnoyLSHIndex(576,100),
    FaissRawIndex(576),
    FaissRawIndex(320),
    FaissRawIndex(192)

]
metadata_info = [
    [["caltech101", "best_resnet18_faisslsh"],
     ["caltech101", "MobileNetV3_small_custom_faisslsh"],
     ["caltech101", "MobileNetV3_large_faisslsh"],
     ["caltech101", "tinyViT_raw"],
     ["caltech101", "TinyViT_small_RawIndex"],
     ["caltech101", "EfficientViT-M0_RawIndex"]],
    [["cifar10", "best_resnet18_faisslsh"],
     ["cifar10", "MobileNetV3_small_custom_faisslsh"],
     ["cifar10", "MobileNetV3_large_faisslsh"]],
    [["oxford5k", "best_resnet18_faisslsh"],
     ["oxford5k", "MobileNetV3_small_custom_faisslsh"],
     ["oxford5k", "MobileNetV3_large_faisslsh"]],
    [["corel5k", "best_resnet18_faisslsh"],
     ["corel5k", "MobileNetV3_small_custom_faisslsh"],
     ["corel5k", "MobileNetV3_large_faisslsh"]],
    [["INRIA_Holiday", "best_resnet18_faisslsh"],
     ["INRIA_Holiday", "MobileNetV3_small_custom_faisslsh"],
     ["INRIA_Holiday", "MobileNetV3_large_faisslsh"],
     ["INRIA_Holiday", "TinyViT_small_RawIndex"],
     ["INRIA_Holiday", "EfficientViT-M0_RawIndex"]]   
]

############################################################################################
################################# CBIR INSTANCE ############################################
############################################################################################
head_output = None

TestSearch = [
    [
        CBIR(BackBoneInstance[5], IndexingInstance[8], metadata=metadata_info[0][5], evalmode=False),
        CBIR(BackBoneInstance[4], IndexingInstance[7], metadata=metadata_info[0][4], evalmode=False),
        CBIR(BackBoneInstance[3], IndexingInstance[6], metadata=metadata_info[0][3], evalmode=False),
        CBIR(BackBoneInstance[0], IndexingInstance[1], metadata=metadata_info[0][0], evalmode=False),
        CBIR(BackBoneInstance[1], IndexingInstance[2], metadata=metadata_info[0][1], evalmode=False),
        CBIR(BackBoneInstance[2], IndexingInstance[3], metadata=metadata_info[0][2], evalmode=False)    
    ],
    [
        CBIR(BackBoneInstance[0], IndexingInstance[1], metadata=metadata_info[1][0], transfer_index = False, evalmode=False),
        CBIR(BackBoneInstance[1], IndexingInstance[2], metadata=metadata_info[1][1], evalmode=False),
        CBIR(BackBoneInstance[2], IndexingInstance[3], metadata=metadata_info[1][2], evalmode=False)    
    ],
    [
        CBIR(BackBoneInstance[0], IndexingInstance[1], metadata=metadata_info[2][0]),
        CBIR(BackBoneInstance[1], IndexingInstance[2], metadata=metadata_info[2][1]),
        CBIR(BackBoneInstance[2], IndexingInstance[3], metadata=metadata_info[2][2])    
    ],
    [
        CBIR(BackBoneInstance[0], IndexingInstance[1], metadata=metadata_info[3][0]),
        CBIR(BackBoneInstance[1], IndexingInstance[2], metadata=metadata_info[3][1]),
        CBIR(BackBoneInstance[2], IndexingInstance[3], metadata=metadata_info[3][2])    
    ],
    [
        CBIR(BackBoneInstance[5], IndexingInstance[8], metadata=metadata_info[4][4], evalmode=False),
        CBIR(BackBoneInstance[4], IndexingInstance[7], metadata=metadata_info[4][3], evalmode=False),
        CBIR(BackBoneInstance[0], IndexingInstance[1], metadata=metadata_info[4][0]),
        CBIR(BackBoneInstance[1], IndexingInstance[2], metadata=metadata_info[4][1]),
        CBIR(BackBoneInstance[2], IndexingInstance[3], metadata=metadata_info[4][2])    
    ]
    
]

############################################################################################
################################# PERFORM INDEXING AND RETRIEVING ##########################
############################################################################################
perform_index = [[True, False, False, False, False, False], 
                 [False, False, False], 
                 [False, False, False], 
                 [False, False, False], 
                 [True, True, False, False, False]]
perform_eval =  [[True, False, False, False, False, False], 
                 [False, False, False], 
                 [False, False, False], 
                 [False, False, False], 
                 [True, True, False, False, False]]
k_top = [5,7,9,11]

for idx_db in range(0, len(TestSearch), 1):
    print("================================= Database",idx_db, "=================================")
    for idx_cbir in range(0, len(TestSearch[idx_db]), 1):
        if perform_index[idx_db][idx_cbir] == True:
            TestSearch[idx_db][idx_cbir].indexDB(mydataloader[idx_db][0])
        # Evaluate phase
        if perform_eval[idx_db][idx_cbir] == True:
            for k in k_top:
                TestSearch[idx_db][idx_cbir].evalRetrieval(mydataloader[idx_db][1], k)
        print("-------------------------------------------------------------------------------")