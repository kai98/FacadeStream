import streamlit as st
import numpy as np
import torch
from PIL import Image
from sources.utils import *

st.set_page_config(page_title='Facade Segmentation', page_icon='🏠', initial_sidebar_state='expanded')

st.title('DeepLab Facade')
uploaded_files = st.sidebar.file_uploader('Upload Facade Images', ['png', 'jpg'], accept_multiple_files=True)

is_save_result = True

filename_list = []


def name_without_extension(name):
    return str(name).split('.')[0]

# Show uploaded_images on the side bar
for uploaded_file in uploaded_files:
    # png image might have the 4th channel - alpha channel.
    img = Image.open(uploaded_file)
    name = uploaded_file.name
    st.sidebar.image(img, caption=name)
    filename_list.append(name_without_extension(name))


def save_uploaded_images(input_path):
    for ufile in uploaded_files:
        _img = Image.open(ufile)
        _img = _img.save(input_path + '/' + ufile.name)
    return


def displayAllPredictions(image_folder):
    for i, fn in enumerate(filename_list):
        displayPrediction(fn, image_folder)
    return


def displayPrediction(filename, _img, _pred, _anno, _wwr):
    cols = st.beta_columns(3)
    cols[0].image(_img, use_column_width=True, caption='Image: ' + filename)
    cols[1].image(_pred, use_column_width=True, caption='prediction')
    cols[2].image(_anno, use_column_width=True, caption='Annotation')

    wwr_percentage = str(round(_wwr * 100, 2)) + "%"

    # Some markdown
    st.markdown("> Window-to-Wall Ratio Estimation:  " + "**" + wwr_percentage + "**")
    st.markdown("------")
    return

def run_prediction():
    prediction_path = './prediction'
    create_folder(prediction_path)

    # Predict each image in uploaded_files
    for ufile in uploaded_files:
        img = np.array(Image.open(ufile).convert('RGB'))
        ufile_name = ufile.name
        filename = name_without_extension(ufile_name)

        # prediction, annotated image, and estimated Window-to-Wall Ratio
        pred_img, anno_image, estimated_wwr = predict(model, img, device)

        fn = name_without_extension(ufile.name)
        displayPrediction(fn, img, pred_img, anno_image, estimated_wwr)

        # save result if is_save_result is true
        if (is_save_result):
            save_result(pred_img, anno_image, estimated_wwr, prediction_path, filename)

    return


# Generate a model with selected weight. Load the model before click analysis.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

st.markdown("> Device - " + "**" + str(device) + "**")

model_path = './models/deeplabv3_facade_2k.pth'
analysis_flag = st.button('Run it!')

model = deeplabv3ModelGenerator(model_path, device)

if analysis_flag:
    run_prediction()
