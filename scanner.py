import streamlit as st
from sources.MachineLearningUtils import *
from sources.FrontendUtils import *

# Parameters:
# Saving prediction locally?
is_save_result = False

# Max height and max width: inputs greater than this shape will be resized.
max_height = 750
max_width = 750

st.set_page_config(page_title='Facade Segmentation', page_icon='🏠', initial_sidebar_state='expanded')
st.title('Facade Segmentation')

uploaded_files = st.sidebar.file_uploader('Upload Facade Images', ['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# Hide setting bar and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

image_list, name_list = process_upload_files(uploaded_files, max_height, max_width)
displaySideBarImage(image_list, name_list)

model_path = './models'

devices_map = get_devices()

model_list = get_model_list(model_path)

if (model_list == []):
    # download_flag = st.button('Download the model')
    cols = st.beta_columns(2)
    download_stage = cols[1].markdown('Model not found 😧 Please download the model. 📦')
    if (cols[0].button('Download Model')):
        download_stage.markdown('Downloading... ⏳')
        download_default_model(model_path)
        download_stage.markdown('Downloaded 🎉')
        model_list = get_model_list(model_path)

cols = st.beta_columns(2)
selected_model = cols[0].selectbox('Model', model_list)

cuda_message = ' - CUDA is available' if torch.cuda.is_available() else ''

# [*devices_map] will return a list of dictionary key, a list of devices' name.
device_key = cols[1].selectbox('Device' + cuda_message, [*devices_map])
device_value = devices_map[device_key]

if (model_list != []):
    device = torch.device(device_value)
    analysis_flag = st.button('Run it! ')

    if analysis_flag:
        # Image to device
        img_device_list = []
        for image in image_list:
            # img_device = image.to(device)
            img_device_list.append(image)

        model = deeplabv3ModelGenerator(model_path + "/" + selected_model, device)
        # Turn on model's evaluation mode
        model.eval()
        # turn off auto grad machine
        with torch.no_grad():
            run_prediction(model, device, img_device_list, name_list, is_save_result)