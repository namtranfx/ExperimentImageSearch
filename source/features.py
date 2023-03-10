from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import torch


from source.losses import TripletLoss


# Our base model
class ResDeepFeature:
    def __init__(self, device):
        self._device = device
        self.model = models.resnet18().cpu()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.triplet_loss = TripletLoss()
        self.is_loaded = False
        self.MODEL_PATH = ".\weight\\result_weight_model.pt"
    
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
    def saveDescriptor(self):
        
        torch.save(self.model.state_dict(), self.MODEL_PATH)
        print("Your feature descriptor saved!!")
    def loadDescriptor(self):
        if self.is_loaded == True:
            print("Your feature descriptor is loaded!")
            return
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        self.model.eval()
        print("Loading your descriptor completed!")
        self.is_loaded = True
    def extractFeature(self, img):
        return self.model(img)
