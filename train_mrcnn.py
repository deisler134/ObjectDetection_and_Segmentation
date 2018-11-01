'''
Created on Nov 1, 2018

@author: Deisler
'''
import os
import csv
import random
import pydicom
import numpy as np
import pandas as pd
from skimage import measure
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt

import sys
import math
import cv2
import json
from imgaug import augmenters as iaa
from tqdm import tqdm
import glob
from sklearn.model_selection import KFold

import os

from data_operation import DataOperation



MRCNN_DIR = os.path.abspath('./')
print('mrcnn_model:',MRCNN_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(MRCNN_DIR, 'logs')

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# pre-trained model
COCO_WEIGHTS_PATH=os.path.join(MRCNN_DIR,'mask_rcnn_coco.h5')
print(COCO_WEIGHTS_PATH)

# The following parameters have been selected to reduce running time for demonstration purposes 
# These are not optimal 

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 6 
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    
    TRAIN_ROIS_PER_IMAGE = 200
    
    MAX_GT_INSTANCES = 4
    
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.01
    VALIDATION_STEPS = 100
    
#     RPN_TRAIN_ANCHORS_PER_IMAGE = 16
    STEPS_PER_EPOCH = 200
#     TOP_DOWN_PYRAMID_SIZE = 32
    TRAIN_BN = True


#config mrcnn and display configuration or write configuration txt
#config.display()
config = DetectorConfig()
config.writeConfig2TXT()

class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__(self)
        
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
   
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)



# add data to list    
def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

# create datalist and annotations
def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations
   


# Root directory of the project
ROOT_DIR = os.path.abspath('../../data')
dataset = DataOperation(ROOT_DIR,'stage_2_train_images','stage_2_test_images','stage_2_train_labels.csv')
train_dicom_dir = dataset.trainpath



if dataset.annspath != '':
    anns = dataset.read_annotations(dataset.annspath)

# Original DICOM image size: 1024 x 1024
ORIG_SIZE = 1024

image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)

image_fps = dataset.random_list(image_fps)
image_fps_train, image_fps_val = dataset.datalist2trainandval(image_fps)

# prepare the training dataset
dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()


# Show annotation(s) for a DICOM image 
test_fp = random.choice(image_fps_train)
image_annotations[test_fp]


# prepare the validation dataset
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()


# Image augmentation 
augmentation = iaa.Sequential([
    iaa.OneOf([ ## geometric transform
        iaa.Affine(
            scale={"x": (0.98, 1.02), "y": (0.98, 1.04)},
            translate_percent={"x": (-0.02, 0.02), "y": (-0.04, 0.04)},
            rotate=(-2, 2),
            shear=(-1, 1),
        ),
        iaa.PiecewiseAffine(scale=(0.001, 0.025)),
    ]),
    iaa.OneOf([ ## brightness or contrast
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([ ## blur or sharpen
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])


#show epoch history
def draw_history(history,epochs,epoch_set,lr_updates):
    plt.figure(figsize=(17,5))
    train_params = 'lr:{}'.format(LEARNING_RATE)
    for i in range(len(epoch_set)):
        train_params += ' {}*{}*lr'.format(epoch_set[i],lr_updates[i])
#     train_param = 'lr:{},epoch1:{},epoch2:{},epoch3:{}'.format(LEARNING_RATE,'{}x2*lr'.format(epochs[0]),'{}xlr'.format(epochs[0]),'{}xlr/5'.format(epochs[0]))
    plt.suptitle(train_params)
    
    plt.subplot(131)
    plt.plot(epochs, history["loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Valid loss")
    plt.legend()
    plt.subplot(132)
    plt.plot(epochs, history["mrcnn_class_loss"], label="Train class ce")
    plt.plot(epochs, history["val_mrcnn_class_loss"], label="Valid class ce")
    plt.legend()
    plt.subplot(133)
    plt.plot(epochs, history["mrcnn_bbox_loss"], label="Train box loss")
    plt.plot(epochs, history["val_mrcnn_bbox_loss"], label="Valid box loss")
    plt.legend()
    
    plt.savefig('history.png')

def saveTrainResult(path='/'):
    path = path.rsplit('/',1)
    path = path[0]+'/result'
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv('{}/train_epoch.csv'.format(path))
    plt.savefig('{}/history.png'.format(path))


# Train Mask-RCNN Model 
import warnings 
warnings.filterwarnings("ignore")

model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

print('start training')
#setting learning_rate
LEARNING_RATE=0.002
epoch_set=[5,20,40,60]
lr_updates=[2,1,1/2,1/4]
layers=['heads','all','all','all','all']
augmentations=[None,augmentation,augmentation,augmentation]
COCO_WEIGHTS_PATH=os.path.join(ROOT_DIR,'mask_rcnn_coco.h5')
if layers[0] == 'heads':
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc",
    "mrcnn_mask_deconv", "mrcnn_mask"])
else:
     model.load_weights(COCO_WEIGHTS_PATH, by_name=True)

history = {}

# add keras custom_callbacks
earlystop = keras.callbacks.EarlyStopping(monitor='val_mrcnn_class_loss', min_delta=1e-4, patience=10)
reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_mrcnn_loss', factor = 0.2, patience = 5)
custom_callbacks = [earlystop, reduce]

for i in range(len(epoch_set)):
    
    model.train(dataset_train, dataset_val, 
            learning_rate=LEARNING_RATE*lr_updates[i], 
            epochs=epoch_set[i], 
            layers=layers[i],
            augmentation=augmentations[i])
    if i == 0:
        history = model.keras_model.history.history
    else:
        new_history = model.keras_model.history.history
        for k in new_history: history[k] = history[k] + new_history[k]
    
    epochs = range(1,len(next(iter(history.values())))+1)
    df=pd.DataFrame(history, index=epochs)
    df.to_csv('train_epoch.csv')
    draw_history(history,epochs,epoch_set,lr_updates)
    saveTrainResult(COCO_WEIGHTS_PATH)




    


