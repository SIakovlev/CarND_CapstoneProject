############################################################################### 
#   Udacity self-driving car course : Capstone Project.
#
#   Author : Charlie Wartnaby, Applus IDIADA
#   Email  : charlie.wartnaby@idiada.com
#
#   Just invokes the CNN model builder so that it downloads the large
#   pretrained model files for later use
############################################################################### 

import tensorflow as tf
import sys
sys.path.insert(0,"../../../../training")
import cnn_classifier_model
import os

print (os.getcwd())

with tf.Session() as sess:
    cnn_model = cnn_classifier_model.CnnClassifierModel(sess, True)
    
