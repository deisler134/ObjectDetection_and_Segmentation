'''
Created on Oct 10, 2018

@author: Deisler
'''

# import os
# import sys
# 
# # Root directory of the project
# ROOT_DIR = os.path.abspath('../../data')
# 
# print(ROOT_DIR)
# 
# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
# 
# if not os.path.exists(ROOT_DIR):
#     os.makedirs(ROOT_DIR)
# os.chdir(ROOT_DIR)
# 
# 
# #Import Mask RCNN
# sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library

from mrcnn.config import Config

class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
        # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 8 
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    
    RPN_ANCHOR_SCALES = (16, 32, 64, 128)
    
    TRAIN_ROIS_PER_IMAGE = 64
    
    MAX_GT_INSTANCES = 4
    
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.78
    DETECTION_NMS_THRESHOLD = 0.01
    
#     RPN_TRAIN_ANCHORS_PER_IMAGE = 16
    STEPS_PER_EPOCH = 200 
#     TOP_DOWN_PYRAMID_SIZE = 32
