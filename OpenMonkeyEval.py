get_ipython().run_line_magic('matplotlib', 'inline')
import json
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay



sp = ['Barbary_macaque', 'Bonobo', 'Orangutan', 'Chacma_baboon',
      'Chimpanzee', 'Common_marmoset', 'Cotton-top_tamarin', 
      'Crab-eating_macaque', 'Gorilla','Emperor_tamarin','Golden_lion_tamarin',
      'Hamadryas_baboon','Japanese_macaque', 'Gibbon',
      'Lion-tailed_macaque','Mandrill','Olive_baboon','Proboscis_monkey',
      'Rhesus_macaque','Siamang','Vervet_monkey','Formosan_rock_macaque',
      'Dusky_leaf_monkey','Golden_snub-nosed_monkey','Tufted_capuchin']



class OpenMonkeyEval:
    def __init__(self, annotation_file=None, submission_file=None):
        self.annotation, self.ann_specs, self.ann_files = dict(),[],[]
        self.submission, self.sub_specs, self.sub_files = dict(),[],[]
        
        if not annotation_file == None:
            data = json.load(open(annotation_file, 'r'))
            assert type(data)==dict, 'annotation file format {} not supported'.format(type(data))
            self.annotation = data
            if 'annotation' in self.annotation:
                for i in range(len(self.annotation['annotation'])):
                    self.ann_specs.append(self.annotation['annotation'][i]['species_id'])
                    self.ann_files.append(self.annotation['annotation'][i]['file_name'])
        else:
            print('Missing annotation file')
                
        if not submission_file == None:
            data = json.load(open(submission_file, 'r'))
            assert type(data)==dict, 'submission file format {} not supported'.format(type(data))
            self.submission = data
            if 'annotation' in self.submission:
                for i in range(len(self.submission['annotation'])):
                    self.sub_specs.append(self.submission['annotation'][i]['species_id'])
                    self.sub_files.append(self.submission['annotation'][i]['file_name'])
        else:
            print('Missing submission file')                    
                    
    def LoadData(self):
        diff = set(self.ann_files) - set(self.sub_files)
        if len(diff) != 0:
            print("Couldn't find the predictions for: {}".format(diff))
        temp = []
        for i in range(len(self.ann_files)):
            try:
                idx = self.sub_files.index(self.ann_files[i])
                temp.append(self.sub_specs[idx])
            except:
                temp.append(0)
        self.sub_specs = temp
        print('There are {} instances in the annotation file.'.format(len(self.ann_files)))
        print('There are {} instances in the submission file.'.format(len(self.ann_files)-len(diff)))
           
        
    def ClassificationReport(self):
        print(classification_report(self.ann_specs, self.sub_specs, target_names=sp))


    def ConfusionMatrix(self):
        cm = confusion_matrix(self.ann_specs, self.sub_specs, labels=range(len(sp)))
        cmp = ConfusionMatrixDisplay(cm, range(len(sp)))
        fig, ax = plt.subplots(figsize=(9,9))
        cmp.plot(ax=ax)

                    
                    
                    
                    
                    
                    
                    
