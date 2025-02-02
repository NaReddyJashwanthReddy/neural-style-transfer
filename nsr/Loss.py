import torch
from logger import logging 

class NSTLoss:
    def __init__(self):
        pass 

    def ContentLoss(self,content_features,target_features):
        return torch.mean((content_features-target_features)**2)
    
    def gram_matrix(self,tensor):
        _,c,h,w=tensor.size()
        logging.info(f"{tensor.size()}")
        tensor=tensor.view(c,h*w)
        logging.info(f"{tensor.size()}")
        return torch.mm(tensor,tensor.t())
    
    def StyleLoss(self,style_feature,target_feature):
        style_gram=self.gram_matrix(style_feature)
        target_gram=self.gram_matrix(target_feature)
        return torch.mean((style_gram-target_gram)**2)
    
    def total_variable_loss(self,image):
        return torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))