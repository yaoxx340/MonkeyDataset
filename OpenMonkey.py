#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import json
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import sys


# In[2]:


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def SpecIdtoName(ids):
    if ids == 0:
        return "Barbary macaque"
    elif ids == 1:
        return "Bonobo"
    elif ids == 2:
        return "Orangutan"    
    elif ids == 3:
        return "Chacma baboon"
    elif ids == 4:
        return "Chimpanzee"
    elif ids == 5:
        return "Common marmoset"
    elif ids == 6:
        return "Cotton-top tamarin"
    elif ids == 7:
        return "Crab-eating_macaque"
    elif ids == 8:
        return "Gibbon"
    elif ids == 9:
        return "Gorilla"
    elif ids == 10:
        return "Dusky leaf monkey"
    elif ids == 11:
        return "Emperor tamarin"
    elif ids == 12:
        return "Formosan rock macaque"
    elif ids == 13:
        return "Golden lion tamarin"
    elif ids == 14:
        return "Golden snub nosed monkey"
    elif ids == 15:
        return "Hamadryas baboon"
    elif ids == 16:
        return "Japanese macaque"
    elif ids == 17:
        return "Lion-tailed macaque"
    elif ids == 18:
        return "Mandrill"
    elif ids == 19:
        return "Olive baboon"
    elif ids == 20:
        return "Proboscis monkey"
    elif ids == 21:
        return "Rhesus macaque"
    elif ids == 22:
        return "Siamang"
    elif ids == 23:
        return "Squirrel monkey"   
    elif ids == 24:
        return "Tufted capuchin"
    elif ids == 25:
        return "Vervet monkey"
    else:
        return -1
    
#REye-LEye-Nose-Head-Neck-RShoulder-RElbow-RHand-LShoulder-LElbow-LHand-Hip-RKnee-RFoot-LKnee-LFoot-Tail
#  0   1    2    3    4       5       6      7      8        9      10   11   12    13   14     15   16
colors = [
            (255, 153, 204),  # REye-Nose
            (255, 153, 204),  # LEye-Nose    
            (153, 51, 255),  # nose-head
            (51, 51, 255),  # head-neck
            (204, 102, 0),  # neck-RShoulder
            (230, 140, 61),  # RShoulder-RElbow
            (255, 178, 102),  # RElbow-RHand
            (255, 102, 102),  # neck-LShoulder
            (255, 179, 102),  # LShoulder-LElbow
            (255, 255, 102),  # LElbow-LHand
            (51, 153, 255),  # neck-hip
            (102, 204, 0),  # hip-RKnee
            (204, 255, 153),  # RKnee-RFoot
            (0, 204, 102),  # hip-LKnee
            (102, 255, 178),  # LKnee-LFoot
            (102, 255, 255),  # hip-tail
        ]
I   = np.array([1,2,3,4,5,6,4,8,9,4,11,12,11,14,11]) 
J   = np.array([2,0,4,5,6,7,8,9,10,11,12,13,14,15,16])


# In[71]:


class OpenMonkey:
    def __init__(self, annotation_file=None, root=None):
        # load dataset
        self.dataset,self.anns,self.specs,self.imgs = dict(),dict(),dict(),dict()
        self.root = root
        if not annotation_file == None:
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            self.dataset = dataset
            self.createIndex()
            print('Annotations loaded.')
            
    def createIndex(self):
        # create index
        anns, specs, imgs, bbox = {}, {}, {}, {}
        if 'annotation' in self.dataset:
            for ann in self.dataset['annotation']:
                anns[ann['id']] = ann['Keypoints']
                imgs[ann['id']] = ann['file_name']
                specs[ann['id']] = ann['species_id']
                bbox[ann['id']] = ann['bbox']


        # create class members
        self.anns = anns
        #self.imgToAnns = imgToAnns
        #self.catToImgs = catToImgs
        self.imgs = imgs
        self.specs = specs
        self.bbox = bbox
    
    def loadImgs(self, ids=[]):
        if _isArrayLike(ids):
            return [self.imgs[i] for i in ids]
        elif type(ids) == int:
            return [self.imgs[ids]] 
        
    def loadAnns(self, ids=[]):
        if _isArrayLike(ids):
            return [self.anns[i] for i in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadSpecs(self, ids=[]):
        if _isArrayLike(ids):
            return [self.specs[i] for i in ids]
        elif type(ids) == int:
            return [self.specs[ids]]

    def loadSpecsNames(self, ids=[]):
        if _isArrayLike(ids):
            return [SpecIdtoName(self.specs[i]) for i in ids]
        elif type(ids) == int:
            return [SpecIdtoName(self.specs[ids])]

    def loadBbox(self, ids=[]):
        if _isArrayLike(ids):
            return [self.bbox[i] for i in ids]
        elif type(ids) == int:
            return [self.bbox[ids]]
    
    def showImgs(self, imgs):
        for i in range(len(imgs)):
            img = cv2.imread(os.path.join(self.root, imgs[i]))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
    def showBbox(self, imgs, bboxs):
        for i in range(len(imgs)):
            img = cv2.imread(os.path.join(self.root, imgs[i]))
            x1 = bboxs[i][0]
            y1 = bboxs[i][1]
            x2 = x1 + bboxs[i][2]
            y2 = y1 + bboxs[i][3]
            img = cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0), 2)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
    def showAnns(self, imgs, anns, bboxs=None, keypoints=False):
        for i in range(len(imgs)):
            img = cv2.imread(os.path.join(self.root, imgs[i]))
            if bboxs != None:
                x1 = bboxs[i][0]
                y1 = bboxs[i][1]
                x2 = x1 + bboxs[i][2]
                y2 = y1 + bboxs[i][3]
                img = cv2.rectangle(img, (x1,y1), (x2,y2),(0,255,0), 2)
            x = anns[i][::3]
            y = anns[i][1::3]
            for j in range(len(I)):
                cv2.line(img,(int(round(x[I[j]])),int(round(y[I[j]]))),(int(round(x[J[j]])),int(round(y[J[j]]))),colors[j], 2)
            if keypoints == True:
                for j in range(len(x)):
                    cv2.circle(img, (int(round(x[j])),int(round(y[j]))), 5, (255,255,255), -1)
                    cv2.circle(img, (int(round(x[j])),int(round(y[j]))), 3, (0,0,0), -1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

    def croppedImgs(self, imgs, bboxs, display=True):
        cropped = []
        for i in range(len(imgs)):
            img = cv2.imread(os.path.join(self.root, imgs[i]))
            x1 = bboxs[i][0]
            y1 = bboxs[i][1]
            x2 = x1 + bboxs[i][2]
            y2 = y1 + bboxs[i][3]
            img = img[y1:y2, x1:x2,:]
            cropped.append(img)
            if display == True:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
        return cropped
           
        
    def croppedAnns(self, imgs, anns, bboxs, display=True):
        cropped = []
        for i in range(len(imgs)):
            img = cv2.imread(os.path.join(self.root, imgs[i]))
            x1 = bboxs[i][0]
            y1 = bboxs[i][1]
            x2 = x1 + bboxs[i][2]
            y2 = y1 + bboxs[i][3]
            img = img[y1:y2, x1:x2,:]
            x = anns[i][::3]
            y = anns[i][1::3]
            x = [j - x1 for j in x]
            y = [j - y1 for j in y]
            for j in range(len(I)):
                cv2.line(img,(int(round(x[I[j]])),int(round(y[I[j]]))),(int(round(x[J[j]])),int(round(y[J[j]]))),colors[j], 2)
            cropped.append(img)
            if display == True:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
        return cropped, [x, y]


# In[72]:







