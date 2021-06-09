import streamlit as st
from PIL import Image
from sources.utils import *
# import numpy as np
# import torch

st.set_page_config(page_title='Facade Segmentation', page_icon='🏠', initial_sidebar_state='expanded')

st.title('DeepLab Facade')
uploaded_files = st.sidebar.file_uploader('Upload Facade Images', ['png', 'jpg'], accept_multiple_files=True)

# Hide setting bar and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


is_save_result = True
filename_list = []

def name_without_extension(filename):
    return ".".join(str(filename).split('.')[:-1])

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

    # Display in 3 columns.
    # col0: original image,
    # col1: prediction (colormap),
    # col0: prediction (annotation),

    cols = st.beta_columns(3)
    cols[0].image(_img, use_column_width=True, caption='Image: ' + filename)
    cols[1].image(_pred, use_column_width=True, caption='prediction')
    cols[2].image(_anno, use_column_width=True, caption='Annotation')

    wwr_percentage = str(round(_wwr * 100, 2)) + "%"

    # Markdown
    st.markdown("> Estimated Window-to-Wall Ratio:  " + "**" + wwr_percentage + "**")
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

        # predict prediction, annotated image, and estimated Window-to-Wall Ratio
        pred_img, anno_image, estimated_wwr = predict(model, img, device)

        fn = name_without_extension(ufile.name)
        displayPrediction(fn, img, pred_img, anno_image, estimated_wwr)

        # save prediction if is_save_result is true
        if (is_save_result):
            save_result(pred_img, anno_image, estimated_wwr, prediction_path, filename)

    return


# The following code: set up device (cuda or not) and select the model.

# CPU / GPU

model_path = './models'

# -----

# Get all cuda devices...Usually just one CPU or one GPU.
def get_devices():
    devices = []
    # Cuda devices
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        devices.append(device_name)

    # CPU option
    devices.append('cpu')
    return devices

device_list = get_devices()
model_list = []

# Scan model_path
for model_name in os.listdir(model_path):
    extension = model_name.split('.')[-1]
    if extension == 'pt' or extension == 'pth':
        model_list.append(model_name)

cols = st.beta_columns(2)

selected_model = cols[0].selectbox('Model', model_list)
selected_device = cols[1].selectbox('Device', device_list)

device = torch.device(selected_device)
analysis_flag = st.button('Run it!')

model = deeplabv3ModelGenerator(model_path + "/" + selected_model, device)

if analysis_flag:
    run_prediction()