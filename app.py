import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from segmentation_models_pytorch import Unet
import numpy as np
import matplotlib.pyplot as plt

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the segmentation model
model = Unet('resnet50', encoder_weights=None, classes=1).to(device)
model.load_state_dict(torch.load('/Users/mitanshpatel/PycharmProjects/pythonstreamlitp1/micronet_resnet50_steel_dataset.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Define preprocessing transformation
def preprocess_image(image):
    # Convert image to RGB (if RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to expected input size
        transforms.ToTensor(),           # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])

    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor

# Define function for segmentation
def segment_image(image, model):
    # Preprocess the image
    image_tensor = preprocess_image(image).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Convert output to numpy array
    segmented_image = output.squeeze().detach().cpu().numpy()

    return segmented_image

# Function to apply custom colormap 'PuRd_r'
def apply_PuRd_r_colormap(segmented_image):
    # Apply PuRd_r colormap to the segmented image
    cmap = plt.get_cmap('PuRd_r')
    colored_image = cmap(segmented_image)[:, :, :3]  # Ignore alpha channel if present
    colored_image *= 255  # Scale colormap values to 0-255 range
    colored_image = colored_image.astype(np.uint8)
    return colored_image

# Streamlit app
st.title('Image Segmentation')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from segmentation_models_pytorch import Unet
import numpy as np
import matplotlib.pyplot as plt

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the segmentation model
model = Unet('resnet50', encoder_weights=None, classes=1).to(device)
model.load_state_dict(torch.load('/Users/mitanshpatel/PycharmProjects/pythonstreamlitp1/micronet_resnet50_steel_dataset.pth', map_location=device))
model.eval()  # Set model to evaluation mode

# Define preprocessing transformation
def preprocess_image(image):
    # Convert image to RGB (if RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to expected input size
        transforms.ToTensor(),           # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet statistics
    ])

    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor

# Define function for segmentation
def segment_image(image, model):
    # Preprocess the image
    image_tensor = preprocess_image(image).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Convert output to numpy array
    segmented_image = output.squeeze().detach().cpu().numpy()

    return segmented_image

# Function to apply custom colormap 'PuRd_r'
def apply_PuRd_r_colormap(segmented_image):
    # Apply PuRd_r colormap to the segmented image
    cmap = plt.get_cmap('PuRd_r')
    colored_image = cmap(segmented_image)[:, :, :3]  # Ignore alpha channel if present
    colored_image *= 255  # Scale colormap values to 0-255 range
    colored_image = colored_image.astype(np.uint8)
    return colored_image

# Streamlit app
st.title('Image Segmentation')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    # Check if the "Proceed" button is clicked
    if st.button('Proceed'):
        # Perform segmentation
        segmented_image = segment_image(image, model)

        # Apply PuRd_r colormap to segmented image
        colored_image = apply_PuRd_r_colormap(segmented_image)

        # Display segmented image with PuRd_r colormap
        st.image(colored_image, caption='Segmented Image (PuRd_r Colormap)', use_column_width=True)


    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    # Check if the "Proceed" button is clicked
    if st.button('Proceed'):
        # Perform segmentation
        segmented_image = segment_image(image, model)

        # Apply PuRd_r colormap to segmented image
        colored_image = apply_PuRd_r_colormap(segmented_image)

        # Display segmented image with PuRd_r colormap
        st.image(colored_image, caption='Segmented Image (PuRd_r Colormap)', use_column_width=True)

