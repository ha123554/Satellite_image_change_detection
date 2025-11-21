# predict.py
import torch
import numpy as np
import cv2

def preprocess(before_img, after_img):
    """
    before_img, after_img: numpy arrays in [0,1], shape HxWx3
    returns: tensor of shape 1x6xHxW
    """
    img = np.concatenate([before_img, after_img], axis=2)  # HxWx6
    tensor = torch.tensor(img.transpose(2,0,1)).unsqueeze(0).float()
    return tensor

def postprocess(pred, threshold=0.2):
    """
    pred: numpy array, predicted change map [0,1]
    returns: binary change map with morphological clean
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    pred_clean = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    return pred_clean

def predict_change(model, before_img, after_img, device='cpu', threshold=0.2):
    model.eval()
    input_tensor = preprocess(before_img, after_img).to(device)
    with torch.no_grad():
        pred = model(input_tensor)[0,0].cpu().numpy()
    pred_binary = postprocess(pred, threshold)
    return pred, pred_binary
