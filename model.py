import streamlit as st
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

from PIL import Image
import numpy as np
import io
import itertools

torch.manual_seed(123)
device='cpu'
input_shape = (3,200,200)
c,img_height,img_width = input_shape
from torchvision.utils import make_grid

# from torchinfo import summary
Tensor=torch.Tensor
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        # Pad --> Conv2d (Same) --> IntanceNorm2d --> Relu --> Pad --> Conv2d (Same)  --> IntanceNorm2d
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
        return x + self.block(x) # Adds the shortcut/bypass connection

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

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)



        self.model = nn.Sequential(
            *self.discriminator_block(channels, 64, normalize=False),
            *self.discriminator_block(64, 128),
            *self.discriminator_block(128, 256),
            *self.discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), # 
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def discriminator_block(self,in_filters, out_filters, normalize=True):
        """Returns downsampling layers of each discriminator block"""
        layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, img):
        return self.model(img)
        
def load_models():
    G_AB = GeneratorResNet(input_shape, num_residual_blocks=3)
    G_BA = GeneratorResNet(input_shape, num_residual_blocks=3)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    G_AB.load_state_dict(torch.load("G_AB_50.pth", map_location=device), strict=False)
    G_BA.load_state_dict(torch.load("G_BA_50.pth", map_location=device), strict=False)
    D_A.load_state_dict(torch.load("D_A_50.pth", map_location=device), strict=False)
    D_B.load_state_dict(torch.load("D_B_50.pth", map_location=device), strict=False)
    # G_AB=torch.load("G_AB_80.pth", map_location=device)
    # G_BA=torch.load("G_BA_80.pth", map_location=device)
    # D_A=torch.load("D_A_80.pth", map_location=device)
    # D_B=torch.load("D_B_80.pth", map_location=device)

    G_AB.eval()
    G_BA.eval()
    D_A.eval()
    D_B.eval()

    return G_AB, G_BA, D_A, D_B
    
G_AB, G_BA, D_A, D_B = load_models()




def generate_image(model, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        fake_image = model(image_tensor)
    return fake_image

def save_image_tensor(image_tensor, file_name):
    image_tensor = image_tensor.squeeze().cpu().detach()
    image = transforms.ToPILImage()(image_tensor)
    image.save(file_name)

transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


class CycleGANDataset2(Dataset):
    def __init__(self, uploaded_file, transforms_=None):
        self.transform = transforms.Compose(transforms_)
    def __getitem__(self, index):
        image_A = Image.open(uploaded_file)
        image_B = Image.open(uploaded_file)


        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return 1
    

def sample_images(type):
    imgs = next(iter(test_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    fake_A = make_grid(fake_A, nrow=1, normalize=True)
    fake_B = make_grid(fake_B, nrow=1, normalize=True)
    if type== 1:
        return fake_B
    else:
        return fake_A
    


st.title('CycleGAN Age Transformation')
st.write('Upload an image to transform it to an aged version.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    test_dataloader = DataLoader(
    CycleGANDataset2(uploaded_file, transforms_=transforms_),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

    fake_image_tensor = sample_images(1)
    save_image_tensor(fake_image_tensor, 'generated_image.jpg')
    st.image('generated_image.jpg', caption='Aged Image', use_column_width=True)




# import streamlit as st
# import torch
# from torchvision import transforms
# import torch.nn as nn
# from torchvision.utils import save_image
# from PIL import Image
# import numpy as np
# import io
# import itertools
# import os
# from PIL import ImageFilter
# import matplotlib.pyplot as plt

# input_shape = (3,40,40)
# c,img_height,img_width = input_shape

# class ResidualBlock(nn.Module):
#     def __init__(self, in_features):
#         super(ResidualBlock, self).__init__()
#         # Pad --> Conv2d (Same) --> IntanceNorm2d --> Relu --> Pad --> Conv2d (Same)  --> IntanceNorm2d
#         self.block = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(in_features, in_features, 3),
#             nn.InstanceNorm2d(in_features),
#             nn.ReLU(inplace=True),
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(in_features, in_features, 3),
#             nn.InstanceNorm2d(in_features),
#         )

#     def forward(self, x):
#         return x + self.block(x)

# class GeneratorResNet(nn.Module):
#     def __init__(self, input_shape, num_residual_blocks=9):
#         '''
#         input_shape : Tensor in (C,H,W) Format
#         num_residual_blocks : Number of Residual blocks
#         '''
#         super(GeneratorResNet, self).__init__()

#         channels = input_shape[0]

#         # Initial convolution block
#         out_features = 64
#         model = [
#             nn.ReflectionPad2d(channels),
#             nn.Conv2d(channels, out_features, 7),
#             nn.InstanceNorm2d(out_features),
#             nn.ReLU(inplace=True),
#         ]
#         in_features = out_features

#         # Downsampling
#         for _ in range(2):
#             out_features *= 2
#             model += [
#                 nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features = out_features

#         # Residual blocks
#         for _ in range(num_residual_blocks):
#             model += [ResidualBlock(out_features)]

#         # Upsampling
#         for _ in range(2):
#             out_features //= 2
#             model += [
#                 nn.Upsample(scale_factor=2),
#                 nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
#                 nn.InstanceNorm2d(out_features),
#                 nn.ReLU(inplace=True),
#             ]
#             in_features = out_features

#         # Output layer
#         model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

#         self.model = nn.Sequential(*model) #Build Sequential model by list unpacking

#     def forward(self, x):
#         return self.model(x)
        
# class Discriminator(nn.Module):
#     def __init__(self, input_shape):
#         super(Discriminator, self).__init__()

#         channels, height, width = input_shape

#         # Calculate output shape of image discriminator (PatchGAN)
#         self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)



#         self.model = nn.Sequential(
#             *self.discriminator_block(channels, 64, normalize=False),
#             *self.discriminator_block(64, 128),
#             *self.discriminator_block(128, 256),
#             *self.discriminator_block(256, 512),
#             nn.ZeroPad2d((1, 0, 1, 0)), # 
#             nn.Conv2d(512, 1, 4, padding=1)
#         )
    
#     def discriminator_block(self,in_filters, out_filters, normalize=True):
#         """Returns downsampling layers of each discriminator block"""
#         layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#         if normalize:
#             layers.append(nn.InstanceNorm2d(out_filters))
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
#         return layers

#     def forward(self, img):
#         return self.model(img)
        
# class SingleImageDataset(Dataset):
#     def __init__(self, image, transform):
#         self.image = image
#         self.transform = transform

#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         return self.transform(self.image)
        
# # class CycleGANDataset(Dataset):
# #     def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
# #         self.transform = transforms.Compose(transforms_)
# #         self.unaligned = unaligned # used to handle cases when the number of images are not equal

# #         self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
# #         self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

# #     def __getitem__(self, index):
# #         image_A = Image.open(self.files_A[index % len(self.files_A)]) # Safe-indexing such that always withing the range

# #         if self.unaligned:
# #             image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
# #         else:
# #             image_B = Image.open(self.files_B[index % len(self.files_B)])


# #         item_A = self.transform(image_A)
# #         item_B = self.transform(image_B)
# #         return {"A": item_A, "B": item_B}


# # def load_models():
# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     G_AB = GeneratorResNet(input_shape, num_residual_blocks=3)
# #     G_BA = GeneratorResNet(input_shape, num_residual_blocks=3)
# #     D_A = Discriminator(input_shape)
# #     D_B = Discriminator(input_shape)

# #     G_AB.load_state_dict(torch.load("G_AB_99.pth", map_location=device), strict=False)
# #     G_BA.load_state_dict(torch.load("G_BA_99.pth", map_location=device), strict=False)
# #     D_A.load_state_dict(torch.load("D_A_99.pth", map_location=device), strict=False)
# #     D_B.load_state_dict(torch.load("D_B_99.pth", map_location=device), strict=False)

# #     G_AB.eval()
# #     G_BA.eval()
# #     D_A.eval()
# #     D_B.eval()

# #     return G_AB, G_BA, D_A, D_B
    
# # G_AB, G_BA, D_A, D_B = load_models()

# # def transform_image(image):
# #     transform = transforms.Compose([
# #         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
# #         transforms.ToTensor(),
# #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
# #     ])
# #     return transform(image).unsqueeze(0)

# # def generate_image(model, image_tensor):
# #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
# #     image_tensor = image_tensor.to(device)
# #     with torch.no_grad():
# #         fake_image = model(image_tensor)
# #     return fake_image
    


# # def sharpen_image(image):
# #     return image.filter(ImageFilter.SHARPEN)

# # def save_image_tensor(image_tensor, file_name):
# #     image_tensor = image_tensor.squeeze().cpu().detach()
# #     image = transforms.ToPILImage()(image_tensor)
# #     image = sharpen_image(image) 
# #     image.save(file_name)

# # st.title('CycleGAN Age Transformation')
# # st.write('Upload an image to transform it to an aged version.')

# # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# # if uploaded_file is not None:
# #     image = Image.open(uploaded_file).convert('RGB')
# #     st.image(image, caption='Uploaded Image', use_column_width=True)

# #     image_tensor = transform_image(image)
# #     # save_image_tensor(image_tensor, 'transformed_image.jpg')
# #     # st.image('transformed_image.jpg', caption='Transformed Image', use_column_width=True)

# #     fake_image_tensor = generate_image(G_AB, image_tensor)
# #     save_image_tensor(fake_image_tensor, 'generated_image.jpg')
# #     st.image('generated_image.jpg', caption='Aged Image', use_column_width=True)



# # Define paths to the models
# # model_paths = {
# #     'G_AB': 'saved_models/your_dataset_name/G_AB_99.pth',
# #     'G_BA': 'saved_models/your_dataset_name/G_BA_99.pth',
# # }

# # Load the models
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Assuming you have the GeneratorResNet class defined as in your previous code
# G_AB = GeneratorResNet(input_shape).to(device)
# G_BA = GeneratorResNet(input_shape).to(device)

# G_AB.load_state_dict(torch.load("G_AB_99.pth", map_location=device), strict=False)
# G_BA.load_state_dict(torch.load("G_BA_99.pth", map_location=device), strict=False)

# G_AB.eval()
# G_BA.eval()

# # Define image transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# # Define inverse transform for displaying images
# inverse_transform = transforms.Compose([
#     transforms.Normalize((-1, -1, -1), (2, 2, 2)),
#     transforms.ToPILImage()
# ])

# # Streamlit UI
# st.title("CycleGAN Image Transformation")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     transformed_image = transform_image(image)
#     dataset = SingleImageDataset(image, transform_image)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     fake_image_tensor = generate_image(G_AB, dataloader)
#     fake_image_tensor = fake_image_tensor.squeeze().cpu().detach()
#     fake_image = transforms.ToPILImage()(fake_image_tensor)

#     st.image(fake_image, caption='Aged Image', use_column_width=True)

