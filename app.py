import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from segmentation_models_pytorch import Unet
import numpy as np
import matplotlib.pyplot as plt
import time
import requests

import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from streamlit import session_state

import os

# Get the base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the model file relative to the base directory
model_path = os.path.join(BASE_DIR, 'micronet_resnet50_steel_dataset.pth')

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the segmentation model
model = Unet('resnet50', encoder_weights=None, classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set model to evaluation mode


# Define preprocessing transformation
def preprocess_image(image):
    # Convert image to RGB (if RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image to expected input size
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Normalize using ImageNet statistics
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


# Function to display README content





# Streamlit app
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_microscope = "https://lottie.host/9f7bd427-e939-44c2-b364-658e6d341c3a/V6NMIbrwcE.json"
lottie_url_2_microscope = "https://lottie.host/7b6fbc4d-49dd-42e6-a63a-e63b30da582a/93kXsc86ns.json"
lottie_url_3_microscope = "https://lottie.host/85ed6aa9-b272-45c9-99ec-7c108a2e086f/dOkI71usy4.json"

lottie_hello = load_lottieurl(lottie_url_microscope)
lottie_hello_1 = load_lottieurl(lottie_url_2_microscope)
lottie_hello_2 = load_lottieurl(lottie_url_3_microscope)

st.markdown('''# Serb_Steel_Segmentation ''')
if lottie_hello:
    st.title('Microscopic Steel Image Segmentation')

import streamlit as st
import time

# Display buttons horizontally
col1, col2, col3 = st.columns(3)

if 'about_project_visible' not in st.session_state:
    st.session_state.about_project_visible = False

if col1.button('About project'):
    st.session_state.about_project_visible = not st.session_state.about_project_visible

if st.session_state.about_project_visible:
    with col1:
        with st_lottie_spinner(lottie_hello_1, key="downloader"):
            time.sleep(1.5)
        st.write("This web application enables users to perform segmentation on microscopic images of steel samples. The segmentation process is powered by a deep learning model implemented using PyTorch and the segmentation_models_pytorch library. The model is a U-Net architecture pretrained on the ResNet50 backbone. This project aims to provide a user-friendly interface for segmenting microscopic images of steel samples, which can be useful in various industrial applications such as quality control and defect detection.")

if 'about_developer_visible' not in st.session_state:
    st.session_state.about_developer_visible = False

if col2.button('About Developer'):
    st.session_state.about_developer_visible = not st.session_state.about_developer_visible

if st.session_state.about_developer_visible:
    with col2:
        with st_lottie_spinner(lottie_hello_1, key="dowloader"):
            time.sleep(1.5)
        st.write("This website is developed by Mitansh Patel and Varun Patel under guidance of Dr. Amitiva Choudary.")

if 'about_model_visible' not in st.session_state:
    st.session_state.about_model_visible = False

if col3.button('About the model'):
    st.session_state.about_model_visible = not st.session_state.about_model_visible

if st.session_state.about_model_visible:
    with col3:
        with st_lottie_spinner(lottie_hello_1, key="downloadr"):
            time.sleep(1.5)
        st.write("The segmentation model used in this web application is based on the U-Net architecture, which is commonly used for image segmentation tasks. The model is pretrained on the ResNet50 backbone to leverage transfer learning and achieve better performance")



uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

up = uploaded_file
if up is not None:

    image = Image.open(up)
    original_image = st.image(image, caption='Original Image', use_column_width=True)

    # Check if the "Segment" button is clicked
    if st.button('Segment'):
        with st_lottie_spinner(lottie_hello, key="downloader"):
            time.sleep(4)
        # Perform segmentation
        segmented_image = segment_image(image, model)

        # Apply PuRd_r colormap to segmented image
        colored_image = apply_PuRd_r_colormap(segmented_image)

        # Display segmented image with PuRd_r colormap
        st.image(colored_image, caption='Segmented Image', use_column_width=True)

    # Add a "Return" button to remove the original image
    if st.button('Return'):
        with st_lottie_spinner(lottie_hello, key="downloaderin"):
            time.sleep(.8)

        original_image.empty()  
        




   

   
       

