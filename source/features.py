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
        print("Training completed!")
    def saveDescriptor(self, path):
        print("Saving your descriptor...")
        torch.save(self.model.state_dict(), path)
    def loadDescriptor(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print("Loading your descriptor completed!")
    def extractFeature(self, img):
        return self.model(img)
