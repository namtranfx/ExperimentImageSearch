from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
from skimage.feature import hog # Import Hog model to extract features
import cv2
from torchvision.models.feature_extraction import create_feature_extractor

from source.losses import TripletLoss
from source.data_handler import MyTransform_norm, MyTransform
import timm 



class FeatureDescriptor:
    def __init__(self) -> None:
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        self.MODEL_PATH = ".\weight\\result_weight_model.pt"
        # self.MODEL_PATH = ".\weight\\resnet18.pth"
        self.model = None
        self.self_trained = False
        self._m_data_transform = MyTransform()
    def trainDescriptor(self):
        pass
    def saveDescriptor(self):
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        print("Your feature descriptor saved!!")
    def loadDescriptor(self):
        pass
    def extractFeature(self, img):
        # img = img.resize((224,224))
        # img = torch.tensor([self._m_data_transform.val_transforms(img).numpy()]).to(device=self._device)
        # img = torch.tensor(np.array([self._m_data_transform.process(img).numpy()])).to(device=self._device)
        
        output = self.model(img)
        # print("[features.py]: shape of output: ", output.shape)
        return output.view(output.size(0), -1) # Flatten the output

    def eval(self):
        if self.model is not None: self.model.eval()
        else: print("[ERROR]: Cannot change model into inference mode")
    def train(self):
        if self.model is not None: self.model.train()
        else: print("[ERROR]: Cannot change model into train mode")
class HOGFeature(FeatureDescriptor):
    def resize_(image):
        u = cv2.resize(image,(224,224))
        return u

    def rgb2gray(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        return gray
    def __init__(self) -> None:
        super().__init__()
        
    def extractFeature(self, img):
        img = self.resize_(img)
        img = self.rgb2gray(img)
        fd = hog(img, orientations=8, pixels_per_cell=(64, 64),block_norm ='L2', cells_per_block=(2, 2))
        return fd

class MobileNetV3Feature(FeatureDescriptor):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).cpu()
        return_nodes = {
            'flatten':'final_feature'
        }
        self.model = create_feature_extractor(self.model, return_nodes=return_nodes)
        # self.model = self.model.features
    def extractFeature(self, img):
        final_f = self.model(img)['final_feature']
        # feature = self.model(img)
        # #print("Shape of feature = ", feature.shape)
        # # return torch.flatten(feature, start_dim=1)
        # compact_f = torch.nn.AdaptiveAvgPool2d(1)(feature)
        # final_f = torch.flatten(compact_f, start_dim=1)
        # # print("[MobileNet feature]: shape of final feature = ", final_f.shape)
        return final_f

class MobileNetV3_small_composite(FeatureDescriptor):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).cpu()
        return_nodes = {
            'flatten' : 'final_feature',
            'classifier.1' : 'first_semantic',
            'classifier.2' : 'final_semantic', 
            'classifier.3' : 'classify_result'
        }
        self.model = create_feature_extractor(self.model, return_nodes=return_nodes)
    def extractFeature(self, img):
        inference_result = self.model(img)
        final_f = inference_result['final_feature'] # shape = 576
        first_semantic = inference_result['first_semantic'] # shape = 1024
        final_classify = inference_result['classify_result'] # shape = 1000
        
        return final_f
class MobileNetV3Feature_flatten(MobileNetV3Feature):
    def extractFeature(self, img):
        feature = self.model.features(img)
        # print("Shape of feature = ", feature.shape)
        # return torch.flatten(feature, start_dim=1)
        # compact_f = torch.nn.AdaptiveAvgPool2d(1)(feature)
        final_f = torch.flatten(feature, start_dim=1)
        #print("shape of final feature = ", final_f.shape)
        return final_f
class MobileNetV3Feature_large(MobileNetV3Feature):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT).cpu()
        return_nodes = {
            'flatten':'final_feature'
        }
        self.model = create_feature_extractor(self.model, return_nodes=return_nodes)
        # self.model = self.model.features
class MobileNetV3Feature_large_flatten(MobileNetV3Feature_large):
    def extractFeature(self, img):
        feature = self.model.features(img)
        # print("Shape of feature = ", feature.shape)
        # return torch.flatten(feature, start_dim=1)
        # compact_f = torch.nn.AdaptiveAvgPool2d(1)(feature)
        final_f = torch.flatten(feature, start_dim=1)
        #print("shape of final feature = ", final_f.shape)
        return final_f

class MobileNetV3_custom(FeatureDescriptor):
    def __init__(self) -> None:
        super().__init__()
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT).cpu()
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer
        
    


# Our base model
class ResDeepFeature(FeatureDescriptor):
    def __init__(self):
        super().__init__()
        self.self_trained = True
        
        # Load model from our self trained model
        model = models.resnet18().cpu() # Full model
        # self.model = torch.nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer

        if self.self_trained == True:
            model.load_state_dict(torch.load(self.MODEL_PATH))
            model.eval()
            print("Your feature descriptor is loaded!")
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer
        
        self.is_loaded = True
        # Use for self trainning model 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.triplet_loss = TripletLoss()
    
    def trainDescriptor(self, train_loader):
        print("Training our Desciptor")
        epochs = 2
        # Training
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            print("Epoch ", epoch)
            for data in tqdm(train_loader):
                #print("cannot go in for loop")
                self.optimizer.zero_grad()
                x1,x2,x3 = data
                e1 = self.model(x1.cpu())
                e2 = self.model(x2.cpu())
                e3 = self.model(x3.cpu()) 
                
                loss = self.triplet_loss(e1,e2,e3)
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
            print("Train Loss: {}".format(epoch_loss.item()))
        self.is_loaded = True
        print("Training completed!")
    # def saveDescriptor(self):
        
    #     torch.save(self.model.state_dict(), self.MODEL_PATH)
    #     print("Your feature descriptor saved!!")
    # def loadDescriptor(self):
    #     if self.is_loaded == True:
    #         print("Your feature descriptor is loaded!")
    #         return
    #     if self.self_trained == True:
    #         self.model.load_state_dict(torch.load(self.MODEL_PATH))
    #         self.model.eval()
    #     print("Your feature descriptor is loaded!")
    #     self.is_loaded = True
    # def extractFeature(self, img):
    #     img = img.resize((224,224))
    #     # img = torch.tensor([self._m_data_transform.val_transforms(img).numpy()]).to(device=self._device)
    #     img = torch.tensor(np.array([self._m_data_transform.val_transforms(img).numpy()])).to(device=self._device)
    #     output = self.model(img)
    #     # print("----shape of output feature: ", output.view(output.size(0), -1).shape)
    #     return output.view(output.size(0), -1) # Flatten the output

from .TinyViT.tiny_vit import TinyViT, tiny_vit_21m_224, tiny_vit_5m_224
class tinyvit(FeatureDescriptor):
    def __init__(self) -> None:
        super().__init__()
        

        # Tạo mô hình TinyViT-21M
        #self.model = TinyViT()
        #pretrained_weights = torch.load('tiny_vit_21m_22kto1k_distill.pth')
        #self.model.load_state_dict(pretrained_weights)
        # Đăng ký forward hook cho stage cuối cùng của mô hình
        #handle = self.model.blocks[-1].register_forward_hook(self.get_last_stage_output)
        self.model = tiny_vit_21m_224(pretrained=True)
    def extractFeature(self, input):
        

        # Đưa dữ liệu qua mô hình
        output = self.model.forward_features(input)
        return output

import torch.nn as nn

from .EfficientViT.build import EfficientViT_M0
class MyEfficientViT(FeatureDescriptor):
    def __init__(self) -> None:
        super().__init__()
        model = EfficientViT_M0(pretrained='efficientvit_m0')
        self.model = nn.Sequential(*list(model.children())[:-1])
    def extractFeature(self, img):
        feature = self.model(img)
        compact_f = torch.nn.AdaptiveAvgPool2d(1)(feature)
        final_f = torch.flatten(compact_f, start_dim=1)
        # print("[EfficientViT feature]: shape of final feature = ", final_f.shape)
        return final_f
class tinyvit_small(tinyvit):
    def __init__(self) -> None:
        self.model = tiny_vit_5m_224(pretrained = True)
    # Định nghĩa một hàm forward hook để lấy giá trị đầu ra của stage cuối cùng
    def get_last_stage_output(module, input, output):
        global last_stage_output
        last_stage_output = output

       
        
class Resnet18_custom_best(FeatureDescriptor):
    """
    This feature use pretrained weight saved in local storage and have the best 
    performance than other pretrained weight
    """
    def __init__(self) -> None:
        super().__init__()
        self.MODEL_PATH = ".\weight\\resnet18.pth"
        # Load model from our self trained model
        model = models.resnet18().cpu() # Full model
        model.load_state_dict(torch.load(self.MODEL_PATH))
        model.eval()
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer
        

class Resnet18Descriptor(FeatureDescriptor):
    """
    Resnet18 feature using downloaded pretrained weight for the model
    """
    def __init__(self) -> None:
        super().__init__()
        model = models.resnet18(models.ResNet18_Weights.DEFAULT).cpu()
        # model = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1).cpu()
        if self.self_trained == True:
            model.load_state_dict(torch.load(self.MODEL_PATH))
            model.eval()
            print("Your feature descriptor is loaded!")
        self.is_loaded = True
        
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer


class Resnet34Descriptor(FeatureDescriptor):
    def __init__(self) -> None:
        super().__init__()
        model = models.resnet34(models.ResNet34_Weights.DEFAULT).cpu()
        # model = models.resnet34(models.ResNet34_Weights.IMAGENET1K_V1).cpu()

        if self.self_trained == True:
            model.load_state_dict(torch.load(self.MODEL_PATH))
            model.eval()
            print("Your feature descriptor is loaded!")
        self.is_loaded = True
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer


class Resnet50Descriptor(FeatureDescriptor):
    def __init__(self, ) -> None:
        super().__init__()
        model = models.resnet50(models.ResNet50_Weights.DEFAULT).cpu()
        # model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V1).cpu()

        if self.self_trained == True:
            model.load_state_dict(torch.load(self.MODEL_PATH))
            model.eval()
            print("Your feature descriptor is loaded!")
        self.is_loaded = True
        self.model = torch.nn.Sequential(*(list(model.children())[:-1])) # strips off last linear layer
        
class SwinTransformer_default(FeatureDescriptor):
    def __init__(self) -> None:
        super().__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True).cpu()
        return_nodes = {
            'head.global_pool.flatten':'final_feature'
        }
        self.only_feature_model = create_feature_extractor(self.model, return_nodes=return_nodes)
        # self.model = self.model.features
    def extractFeature(self, img):
        final_f = self.only_feature_model(img)['final_feature']
        return final_f
