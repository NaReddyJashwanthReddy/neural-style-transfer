from PIL import Image 
import torch
from torchvision import transforms
from logger import logging 

device=torch.device('cuda')
logging.info(f"device : {device}")

class LoadImage:
    def __init__(self):
        pass 

    def Load_Image(self,path,max_size=512):
        image=Image.open(path)
        transform=transforms.Compose([
            transforms.Resize(max_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        image=transform(image).unsqueeze(0)
        return image.to(device)
    
    def ImShow(self,image):
        image=image.cpu().clone().squeeze(0)
        image=image.detach().numpy()
        image=image.transpose(1,2,0)
        return image

    def Denormalize(self,tensor):
        mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
        std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
        tensor=tensor*std+mean 
        return tensor.clamp(0,1)

    
class Features:

    def __init__(self):
        pass
    
    def get_features(self,image,model,layers):
        features={}
        x=image
        for name,layer in model.named_children():
            x=layer(x)
            if name in layers:
                features[name]=x 

        return features
    
