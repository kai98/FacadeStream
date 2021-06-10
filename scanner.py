import streamlit as st
from PIL import Image
from sources.utils import *
import gdown

st.set_page_config(page_title='Facade Segmentation', page_icon='🏠', initial_sidebar_state='expanded')
st.title('DeepLab Facade')

uploaded_files = st.sidebar.file_uploader('Upload Facade Images', ['png', 'jpg', 'jpeg'], accept_multiple_files=True)

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
    cols[1].image(_pred, use_column_width=True, caption='Prediction')
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

def download_default_model(dir):
    google_drive_url = 'https://drive.google.com/uc?export=download&id=1rJ3edeARtcprrgs14lj5iZLTLkn9kufw'
    output = dir + '/deeplabv3_resnet101.pth'
    # md5 = '31aa0b0a3607b091975b5e05df590280'
    gdown.download(google_drive_url, output,  quiet=False)
    # gdown.cached_download(google_drive_url, output, md5=md5)
# -----

# Download model


# Get all cuda devices...Usually just one CPU or one GPU.
def get_devices():
    devices = []
    devices_map = {}
    # Cuda devices
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i) + '_' + str(i)
        devices.append(device_name)
        devices_map[device_name] = 'cuda:' + str(i)

    # CPU option
    cpu_name = 'CPU'
    devices.append(cpu_name)
    devices_map[cpu_name] = 'cpu'
    return devices_map

devices_map = get_devices()


# Scan model_path
def get_model_list (dir):
    models = []
    if not os.path.exists(dir):
        os.mkdir(dir)

    for model_name in os.listdir(dir):
        extension = model_name.split('.')[-1]
        if extension == 'pt' or extension == 'pth':
            models.append(model_name)
    return models

model_list = get_model_list(model_path)

if (model_list == []):
    # download_flag = st.button('Download the model')
    cols = st.beta_columns(2)
    download_stage = cols[1].markdown('Model not found 😧')
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
    model = deeplabv3ModelGenerator(model_path + "/" + selected_model, device)
    # print('Model name: %s'% selected_model)
    # print('Device: %s'%device_value)
    if analysis_flag:
        run_prediction()
