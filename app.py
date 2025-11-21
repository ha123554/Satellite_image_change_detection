# app.py
import streamlit as st
import numpy as np
from PIL import Image
import cv2
from model import UNet
from predict import predict_change
import torch


st.title("Satellite Image Change Detection Demo")

# Upload Before & After Images
before_file = st.file_uploader("Upload BEFORE Image", type=["png","jpg"])
after_file = st.file_uploader("Upload AFTER Image", type=["png","jpg"])

if before_file and after_file:
    before_img = np.array(Image.open(before_file).convert('RGB')) / 255.0
    after_img = np.array(Image.open(after_file).convert('RGB')) / 255.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load("satellite_chnge_detection.pth", map_location=device))
    
    # Predict
    pred, pred_binary = predict_change(model, before_img, after_img, device=device, threshold=0.2)
    
    # Overlay
    after_uint8 = (after_img*255).astype(np.uint8)
    overlay = cv2.addWeighted(after_uint8, 0.7, cv2.applyColorMap(pred_binary*255, cv2.COLORMAP_JET), 0.3, 0)
    
    st.image(before_img, caption="Before Image", use_container_width=True)
    st.image(after_img, caption="After Image", use_container_width=True)
    st.image(pred, caption="Predicted Change Map", use_container_width=True)
    st.image(pred_binary*255, caption="Binary Change Map", use_container_width=True)
    st.image(overlay, caption="Overlay", use_container_width=True)

