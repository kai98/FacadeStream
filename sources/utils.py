from torchvision import transforms
import torchvision
import torch
import numpy as np
import cv2

transforms_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# One image at a time. 
def predict(model, image, filename, prediction_path, device):
    # TODO: Multiple image support.
    # Number of classes in the dataset

    # turn on evaluation 
    model.eval()

    # make sure image is nparray
    image = np.array(image)

    prediction_indexed= label_image(model, image, device)

    prediction_color = decode_segmap(prediction_indexed)

    annotation = annotate_image(image, prediction_indexed)

    # make sure in the right format
    # image = image.astype('uint8')

    # Image channel.. 
    annotation = cv2.cvtColor(annotation, cv2.COLOR_RGB2BGR)
    prediction = cv2.cvtColor(prediction_color, cv2.COLOR_RGB2BGR)

    path_file_name = prediction_path + '/' + filename

    cv2.imwrite(path_file_name + "_annotation.jpg", annotation)
    cv2.imwrite(path_file_name + "_prediction.jpg", prediction)

    # Also return the original predicted value. 
    return get_wwr_by_pixel(prediction_indexed)


def get_wwr_by_pixel(prediction_indexed):

    # 0 Void, or various
    # 1 Wall
    # 2 Car
    # 3 Door
    # 4 Pavement
    # 5 Road
    # 6 Sky
    # 7 Vegetation
    # 8 Windows
    window_count = np.sum(prediction_indexed == 8)
    wall_count = np.sum(prediction_indexed == 1)
    door_count = np.sum(prediction_indexed == 3)

    return window_count / (window_count + wall_count + door_count)


def annotate_image(image, pred_indexed):

    annotate_colors = {
        0 : (0, 0, 0),              # Various
        1 : (128, 0, 0),            # Wall
        2 : (128, 0, 128),          # Car
        3 : (128, 128, 0),          # Door
        4 : (128, 128, 128),        # Pavement
        5 : (128, 64, 0),           # Road
        6 : (0, 128, 128),          # Sky
        7 : (0, 128, 0),            # Vegetation
        8 : (0, 0, 128)             # Windows
    }
    image = np.array(image)
    
    # 
    dim_factor = 0.5
    image = image * dim_factor

    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    anno_factor = 0.5

    for l in annotate_colors:
        idx = pred_indexed == l
        r[idx] += annotate_colors[l][0] * anno_factor
        g[idx] += annotate_colors[l][1] * anno_factor
        b[idx] += annotate_colors[l][2] * anno_factor

    rgb = np.stack([r, g, b], axis=2)
    rgb = rgb.clip(0, 255)
    image = rgb.astype('uint8')
    return image


# Define the helper function
def decode_segmap(pred_indexed, nc=9):


    # 0 Void, or various
    # 1 Wall
    # 2 Car
    # 3 Door
    # 4 Pavement
    # 5 Road
    # 6 Sky
    # 7 Vegetation
    # 8 Windows

    label_colors = {
        0 : (0, 0, 0),              # Various
        1 : (128, 0, 0),            # Wall
        2 : (128, 0, 128),          # Car
        3 : (128, 128, 0),          # Door
        4 : (128, 128, 128),        # Pavement
        5 : (128, 64, 0),           # Road
        6 : (0, 128, 128),          # Sky
        7 : (0, 128, 0),            # Vegetation
        8 : (0, 0, 128)             # Windows
    }


    r = np.zeros_like(pred_indexed).astype(np.uint8)
    g = np.zeros_like(pred_indexed).astype(np.uint8)
    b = np.zeros_like(pred_indexed).astype(np.uint8)
    

    for l in range(0, nc):
        idx = pred_indexed == l
        r[idx] = label_colors[l][0]
        g[idx] = label_colors[l][1]
        b[idx] = label_colors[l][2]
    
    rgb = np.stack([r, g, b], axis=2)
    
    return rgb


def label_image(model, image, device):
    image = transforms_image(image)
    image = image.unsqueeze(0)
    image_np = np.asarray(image)

    image = image.to(device)
    outputs = model(image)["out"]

    _, preds = torch.max(outputs, 1)

    preds = preds.to("cpu")

    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

    return preds_np

def init_deeplab(num_classes):

    model_deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101()
    model_deeplabv3.aux_classifier = None
    model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)

    return model_deeplabv3