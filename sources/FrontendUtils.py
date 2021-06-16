import streamlit as st
from PIL import Image
from sources.MachineLearningUtils import *
import gdown

def name_without_extension(filename):
    return ".".join(str(filename).split('.')[:-1])

def process_upload_files(uploaded_files, max_height=500, max_width=500):
    image_list = []
    name_list = []

    # process uploaded images
    for up_file in uploaded_files:
        filename = up_file.name
        name_list.append(filename)

        img = np.array(Image.open(up_file).convert('RGB'))
        img = resize_image(img, max_height, max_width)
        image_list.append(img)
    return image_list, name_list

def displaySideBarImage(image_list, name_list):
    input_size = len(image_list)
    for i in range(input_size):
        img = image_list[i]
        name = name_list[i]
        st.sidebar.image(img, caption=name)

def save_uploaded_images(input_path):
    for ufile in uploaded_files:
        _img = Image.open(ufile)
        _img = _img.save(input_path + '/' + ufile.name)
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

def run_prediction(model, device, image_list, name_list, is_save_result=False):
    input_size = len(image_list)

    for i in range(input_size):
        img = image_list[i]
        fn = name_without_extension(name_list[i])

        # predict prediction, annotated image, and estimated Window-to-Wall Ratio
        pred_img, anno_image, estimated_wwr = predict(model, img, device)
        displayPrediction(fn, img, pred_img, anno_image, estimated_wwr)

        # save prediction if is_save_result is true
        if (is_save_result):
            save_result(pred_img, anno_image, estimated_wwr, prediction_path, filename)

    return

def download_default_model(dir):
    google_drive_url = 'https://drive.google.com/uc?export=download&id=1rJ3edeARtcprrgs14lj5iZLTLkn9kufw'
    output = dir + '/deeplabv3_resnet101.pth'
    # md5 = '31aa0b0a3607b091975b5e05df590280'
    gdown.download(google_drive_url, output,  quiet=False)
    # gdown.cached_download(google_drive_url, output, md5=md5)

# Get all cuda devices...Usually just one CPU or one GPU.
def get_devices():
    devices = []
    devices_map = {}
    # Cuda devices
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i) + ' [%s]'%str(i)
        devices.append(device_name)
        devices_map[device_name] = 'cuda:' + str(i)

    # CPU option
    cpu_name = 'CPU'
    devices.append(cpu_name)
    devices_map[cpu_name] = 'cpu'
    return devices_map

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

