# SmartCarla. System Integration (Capstone) Project Write-Up

Self-Driving Car Engineer Nanodegree Program

Version: 1.0

Date: 03Aug2018

## Team smart-carla

|      Name  |          Email |    Responsibilities |
|:------------:|:------------:|:-----------:|
| Sergey Iakovlev | siakovlev@studnet.unimelb.edu.au |  Team lead/Faster R-CNN classifier |
| Tatsuya Hatanaka |                              |                                                      |
| Swapan Shridhar  |                              |                                                      |
| Anthony T Tran   |                              |                                                      |
| Charlie Wartnaby | charlie.wartnaby@idiada.com | Auto simulation training image capture/VGG classifier |


## Abstract

[A few sentences saying that we did it, and how.]

## Submission checklist

This section describes how each of the required checklist items have been met
for project submission.

[CW: by having a sentence or two for each of these, we should make it really
easy for the Udacity assessor to see we've done everything we are supposed to.]

**Launch correctly using the launch files provided in the capstone repo**

**Smoothly follow waypoints in the simulator**

**Respect the target top speed set for the waypoints**

**Stop at traffic lights when needed**

**Stop and restart PID controllers depending on the state of /vehicle/dbw_enabled**

**Publish throttle, steering, and brake commands at 50 Hz**

**Test it out using ROS bags that were recorded at the test site**


## Required set-up

[CW: notes on any one-time scripts to run, etc]

## Waypoint processing

## Drive-by-wire controls

### Steering

### Braking and acceleration

## Training image capture

### Simulation images

[CW: will write something here about automatic capture of classified simulation images]

### Real images

## Image classifier and results: Faster R-CNN

### Tensorflow object detection API
One way build a classifier is to use a powerfull API from Tensorflow on Object Detection. It has many pre-trained networks that can be fine tuned using custom dataset. There are several good tutorials that cover main steps and were used as references in our project:
  - by Daniel Stang: [link](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)
  - by Vatsal SodhaL [link](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)
  - by Dat Tran: [link](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)


As a pre-trained classifier, we chose the Faster R-CNN network ([link](https://arxiv.org/pdf/1504.08083.pdf)) that incorporates a ResNet 101 pretrained model. It provides a good balance between speed and detection accuracy for small objects on the image. In particular, we did not choose SSD (Singe Shot Detector) network as it resizes any input image to 300x300 pixels and, therefore, the accuracy of detecting small objects is reduced. The summary of main speed characteristics for different object detectors can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

### Data preparation, model setup and training
- Data was gathered in two different ways:
  - using `rviz` ROS package following the tutorial from ROS webpage: http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data
  - (Charlie's method, show lines in the code where it was implemented)

- Once the data was ready the following steps were taken to get classifier working:
  - Follow installation instructions for object detection api: [link](https://github.com/tensorflow/models/blob/d1173bc9714b5729b8c95d8e91e8647c66acebe6/object_detection/g3doc/installation.md)
  - Draw boxes and give a corresponding label (`red`, `yellow` or `green`) for each image in the dataset. We used [labelImg](https://github.com/tzutalin/labelImg) to do this. 
  - Create `.pbtxt` file with labels data. See [](link)
  - Create `.record` file. To do this we adopted code from the original [`create_pascal_tf_record.py`](https://github.com/tensorflow/models/blob/d1173bc9714b5729b8c95d8e91e8647c66acebe6/object_detection/create_pascal_tf_record.py). The modified version of the script can be found here: [tf_record_udacity.py](link)
    Usage example:
    ```bash
    python tf_record_udacity.py \
          --data_dir=/home/user/data \ # dataset path
          --output_path=/home/user/udacity_data.record \ # output .record file path
          --label_map_path=/home/user/data/label_map.pbtxt # label .pbtxt file path
    ```
  - Configure the model parameters, trainig and validation setting using corresponding `.config` file. See `faster_rcnn_resnet101_udacity.config` for details.
  - Run `train.py` script, specify model configuration file and output directory. Usage example:
    ```bash
    python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=faster_rcnn_resnet101_udacity.config
    ```
  - Run `export_inference_graph.py` script, specify model configuration file, checkpoint file (`.ckpt`) and output directory for a frozen graph. Usage example:
    ```bash
    python export_inference_graph.py --input_type image_tensor \
    --pipeline_config_path ./faster_rcnn_resnet101_udacity.config \
    --trained_checkpoint_prefix ./models/train/model.ckpt-10000 \
    --output_directory ./fine_tuned_model
    ```
    
### Running instructions
- The flag `is_site` (inside `tl_classifier.py` line 16) is used for switching between two types of classifiers: one is based on simulator images and another is a real images classifier. 
- Once code is running the required model is automatically downloaded and configured. The user will see corresponding messages signifying that classifier was set up successfully. To run the code, GPU enabled machine is required. 
P.S. Tests were conducted using Nvidia GTX1070 8GB, i7-7700HQ.

### Issues:
- Back compatibility with Tensorflow 1.3 as required by Udacity. The current version of object detection API is based on 1.9 tensorflow and not compatible with 1.3. Instead we used a one year old version taken from this [commit](https://github.com/tensorflow/models/tree/d1173bc9714b5729b8c95d8e91e8647c66acebe6).

### Results
[Placeholder for my results]

## Image classifier and results: VGG

[CW: will write something here about my adventures with VGG, and include video]

## Other sections I've forgotten about

## Summary

[CW: We did it. Again. ?]
