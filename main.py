import streamlit as st
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2 as cv
import os
from sources.utils import predict
from sources.utils import init_deeplab
import torch
import numpy as np

# st.set_page_config(page_title='Facade Segmentation', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.set_page_config(page_title='Facade Segmentation', initial_sidebar_state = 'expanded')

st.title('DeepLab Facade')
uploaded_files = st.sidebar.file_uploader('Upload Facade Images', ['png', 'jpg'], accept_multiple_files=True)

filename_list = []
wwr_dictionary = {}

def name_without_extension(name):
    return str(name).split('.')[0]

# Show uploaded_images on the side bar
for uploaded_file in uploaded_files:
    # png image might have the 4th channel - alpha channel.
    img = Image.open(uploaded_file)
    name = uploaded_file.name
    plt.imshow(img)
    plt.show()
    st.sidebar.image(img, caption=name)
    filename_list.append(name_without_extension(name))




def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return


# def clear_folder(folder_path):
#     # Check if the image uploading folder exist
#     # If yes, delete all file under this path
#     if os.path.exists(folder_path):
#         for img_name in os.listdir(folder_path):
#             filepath = os.path.join(folder_path, img_name)
#             print(filepath)
#             os.remove(filepath)
#     # if the folder doesn't exist, create an empty folder.
#     else:
#         os.mkdir(folder_path)
#     return

def save_uploaded_images(input_path):
    for ufile in uploaded_files:
        _img = Image.open(ufile)
        _img = _img.save(input_path + '/' + ufile.name)

    return


def displayAllPredictions(image_folder):
    for i, fn in enumerate(filename_list):
        displayPrediction(fn, image_folder)
    return


def displayPrediction(filename, image_folder):
    anno_postfix = '_annotation.jpg'
    pred_postfix = '_prediction.jpg'

    annotation = image_folder + '/' + filename + anno_postfix
    prediction = image_folder + '/' + filename + pred_postfix

    cols = st.beta_columns(2)
    cols[0].image(prediction, use_column_width=True, caption='prediction: ' + filename)
    cols[1].image(annotation, use_column_width=True, caption='Annotation: ' + filename)

    # get WWR
    wwr = wwr_dictionary[filename]
    wwr_percentage = str(round(wwr * 100, 2)) + "%"

    # Some markdown
    st.markdown("> Window-to-Wall Ratio Estimation:  " + "**" + wwr_percentage + "**")
    st.markdown("------")
    return


def deeplabv3ModelGenerator(model_path):
    num_classes = 9
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # st.markdown("> Device: " + str(device))

    model = init_deeplab(num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.load_state_dict(state_dict)

    return model, device


def run_prediction():
    # upload_path = './images/upload'
    # image_path = './images'
    prediction_path = './prediction'

    # Clear both input and output folder. 
    # clear_folder(upload_path)
    # clear_folder(prediction_path)

    # Save images to input folder
    # save_uploaded_images(upload_path)
    # create_folder(image_path)
    create_folder(prediction_path)

    # Make prediction
    for ufile in uploaded_files:
        img = np.array(Image.open(ufile).convert('RGB'))
        ufile_name = ufile.name
        filename = name_without_extension(ufile_name)
        # model, image, filename, prediction_path, device
        wwr_val = predict(deeplabv3_model, img, filename, prediction_path, device)
        wwr_dictionary[filename] = wwr_val

        fn = name_without_extension(ufile.name)
        displayPrediction(fn, prediction_path)

    # Display all prediction and WWR at once
    # displayAllPredictions(prediction_path)
    return


# Generate a model with selected weight. Load the model before click analysis.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.markdown("> Device - " + str(device))

model_path = './models/deeplabv3_facade_2k.pth'
analysis_flag = st.button('Segment!')

deeplabv3_model, device = deeplabv3ModelGenerator(model_path)

if analysis_flag:
    run_prediction()
