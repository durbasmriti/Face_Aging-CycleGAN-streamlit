import streamlit as st
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import io
import itertools
n_epochs=100
epoch=offset=0 

decay_start_epoch=3
input_shape = (3,40,40)
c,img_height,img_width = input_shape
batch_size = 100
lr = 2e-4
checkpoint_interval = 1
sample_interval = 100
lambda_cyc = 10
lambda_id = 5
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
input_shape = (3, 200, 200)
G_AB = GeneratorResNet(input_shape, num_residual_blocks=9)
G_BA = GeneratorResNet(input_shape, num_residual_blocks=9)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_flag = torch.cuda.is_available()

if gpu_flag:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
# Load pre-trained weights
try:
    state_dict_G_AB = torch.load("G_AB_99.pth", map_location=device)
    state_dict_G_BA = torch.load("G_BA_99.pth", map_location=device)
    state_dict_D_A = torch.load("D_A_99.pth", map_location=device)
    state_dict_D_B = torch.load("D_B_99.pth", map_location=device)
    
    # Print model and state dict keys for debugging
    print("Model G_AB keys:")
    print([key for key, _ in G_AB.named_parameters()])
    print("State dict keys for G_AB:")
    print(state_dict_G_AB.keys())
    
    print("Model G_BA keys:")
    print([key for key, _ in G_BA.named_parameters()])
    print("State dict keys for G_BA:")
    print(state_dict_G_BA.keys())
    
    print("Model D_A keys:")
    print([key for key, _ in D_A.named_parameters()])
    print("State dict keys for D_A:")
    print(state_dict_D_A.keys())
    
    print("Model D_B keys:")
    print([key for key, _ in D_B.named_parameters()])
    print("State dict keys for D_B:")
    print(state_dict_D_B.keys())
    
    # Load state dictionaries into models
    G_AB.load_state_dict(state_dict_G_AB)
    G_BA.load_state_dict(state_dict_G_BA)
    D_A.load_state_dict(state_dict_D_A)
    D_B.load_state_dict(state_dict_D_B)
    print("Models loaded successfully!")
except RuntimeError as e:
    print(f"Error loading model weights: {e}")
    
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

# Define transformations
transform_pipeline = transforms.Compose([
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

st.title("Image Aging with CycleGAN")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image_tensor = transform_pipeline(image).unsqueeze(0).to(device)
    with torch.no_grad():
        aged_image_tensor = G_AB(image_tensor)
    
    aged_image = aged_image_tensor.squeeze(0).cpu().detach()
    aged_image = transforms.ToPILImage()(aged_image * 0.5 + 0.5)

    st.image(aged_image, caption='Aged Image', use_column_width=True)

