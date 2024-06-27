import streamlit as st
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import io
import itertools

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
        '''
        input_shape : Tensor in (C,H,W) Format
        num_residual_blocks : Number of Residual blocks
        '''
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model) #Build Sequential model by list unpacking

    def forward(self, x):
        return self.model(x)
        
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
        self.model = nn.Sequential(
            *self.discriminator_block(channels, 64, normalize=False),
            *self.discriminator_block(64, 128),
            *self.discriminator_block(128, 256),
            *self.discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def discriminator_block(self, in_filters, out_filters, normalize=True):
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, img):
        return self.model(img)
        
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_shape = (3, 256, 256)

    G_AB = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
    G_BA = GeneratorResNet(input_shape, num_residual_blocks=9).to(device)
    D_A = Discriminator(input_shape).to(device)
    D_B = Discriminator(input_shape).to(device)

    G_AB.load_state_dict(torch.load("G_AB_99.pth", map_location=device), strict=False)
    G_BA.load_state_dict(torch.load("G_BA_99.pth", map_location=device), strict=False)
    D_A.load_state_dict(torch.load("D_A_99.pth", map_location=device), strict=False)
    D_B.load_state_dict(torch.load("D_B_99.pth", map_location=device), strict=False)

    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()

    return G_AB, G_BA, D_A, D_B


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

def generate_image(model, image_tensor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        fake_image = model(image_tensor)
    return fake_image

st.title('CycleGAN Age Transformation')
st.write('Upload an image to transform it to an aged version.')

G_AB, G_BA, D_A, D_B = load_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_tensor = transform_image(image)
    fake_image_tensor = generate_image(G_AB, image_tensor)
    fake_image_tensor = fake_image_tensor.squeeze().cpu().detach()
    fake_image = transforms.ToPILImage()(fake_image_tensor)

    st.image(fake_image, caption='Aged Image', use_column_width=True)
