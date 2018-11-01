'''
Created on Nov 1, 2018

@author: Deisler
'''

import os
from tqdm import tqdm
import glob
import pandas as pd
import random


############################################################
#  build path and return
############################################################

def build_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

############################################################
#  training data process class
############################################################

class DataOperation():
    """
    Base data Operation class for mrcnn train
    """
    def __init__(self, dataroot, trainfolder = '', testfolder = '', anns = ''):
        
        self.dataroot = build_path(dataroot)        
        self.trainpath = os.path.join(dataroot, trainfolder)
        self.testpath = os.path.join(dataroot, trainfolder)
        self.annspath = os.path.join(dataroot, anns)
        
    
    def get_datalist(self, folderpath = '', fileType = ''):
        
        fps = glob.glob(os.path.join(folderpath, fileType))
        return list(set(fps))
    
    def read_annotations(self, annspath = ''):
        
        if annspath != '':
            self.annspath = annspath
        return pd.read_csv(self.annspath)
    
    def random_list(self,datalist, randomseed = 43):
        
        sorted(datalist)
        random.seed(randomseed)
        random.shuffle(datalist)
        return datalist
    
    def datalist2trainandval(self, datalist, valratio = 0.1):
        
        split_index = int((1 - valratio) * len(datalist))
        trainlist = datalist[:split_index]
        vallist = datalist[split_index:]
        return trainlist, vallist
        
        
    


    
