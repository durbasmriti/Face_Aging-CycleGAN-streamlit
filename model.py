import streamlit as st
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import io

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

input_shape = (3, 256, 256)
transform = transforms.Compose([
    transforms.Resize((input_shape[1], input_shape[2])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_AB = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
G_AB.load_state_dict(torch.load("G_AB_99.pth", map_location=device))

G_BA = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
G_BA.load_state_dict(torch.load("G_BA_99.pth", map_location=device))

st.title("Image Aging with CycleGAN")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        aged_image_tensor = G_AB(image_tensor)
    
    aged_image = aged_image_tensor.squeeze(0).cpu().detach()
    aged_image = transforms.ToPILImage()(aged_image * 0.5 + 0.5)

    st.image(aged_image, caption='Aged Image', use_column_width=True)

