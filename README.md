# SmartCarla. System Integration (Capstone) Project Write-Up

Self-Driving Car Engineer Nanodegree Program

Version: 1.0

Date: 03Aug2018

## Team smart-carla

|      Name  |          Email |    Responsibilities |
|:------------:|:------------:|:-----------:|
| Sergey Iakovlev | siakovlev@studnet.unimelb.edu.au |  Team lead/Faster R-CNN classifier |
| Tatsuya Hatanaka | tatsuya.hatanaka@gmail.com | Data preparation, Smooth braking |
| Swapan Shridhar  | available365.24.7@gmail.com  | Twist Controller, Finalizing the README              |
| Anthony T Tran   |                              |                                                      |
| Charlie Wartnaby | charlie.wartnaby@idiada.com | Auto simulation training image capture/VGG classifier |


## Abstract

This is the project repo of **Smart Carla team** for the final project of the Udacity Self-Driving Car Nanodegree in which we programed a real self-driving car.

The goal of the project was to get **Udacity's self-driving car** to drive around a test track while avoiding obstacles and stopping at traffic lights.

The starting code has been taken from Udacity's github repository [here](https://github.com/udacity/CarND-Capstone).



## Submission checklist

This section describes how each of the required checklist items have been met
for project submission.

[CW: by having a sentence or two for each of these, we should make it really
easy for the Udacity assessor to see we've done everything we are supposed to.]

* [x] **Launch correctly using the launch files provided in the capstone repo**

* [x] **Smoothly follow waypoints in the simulator**

* [x] **Respect the target top speed set for the waypoints**

* [x] **Stop at traffic lights when needed**

* [x] **Stop and restart PID controllers depending on the state of /vehicle/dbw_enabled**

* [x] **Publish throttle, steering, and brake commands at 50 Hz**

* [x] **Test it out using ROS bags that were recorded at the test site**


## Required set-up

[CW: notes on any one-time scripts to run, etc]

## Waypoint processing

## Drive-by-wire controls

### Steering

"wayoint_follower" pure_pursuit_core.h params updated to enable more fine grained check in **verifyFollowing()** to fix cars wandering around the waypoints.

### Braking and acceleration

## Training image capture

Automatic collection of images for classifier training purposes was added to `tl_detector.py`.
This collected images from the `/image_color` topic, either from the simulator (acting as a
virtual camera sensor), or from playing one of the Udacity-provided .bag files with real
images.

The images were captured if `self.grab_training_images` (normally False) was set True. The
files were automatically named and numbered (using `self.training_image_idx`). However, The
strategy for collecting simulator or real images then differed.

The end result was a collection of images named e.g. `sim_123_0.jpg` (for the 123rd simulator
image of state 0=RED) or `real_124_2.jpg` (for the 124th real image of state 2=GREEN). These
images could then be read into the training programs directly and the ground truth state
extracted easily from the filename suffix. The training images can be found in the 
`data\training_images*` folders.
 
### Simulation images

The simulator provided ground truth light states (colours) alongside the images, so
we wrote code in `tl_detector.py` to automatically name the saved image files with the
required ground-truth suffix number, requiring no manual work.

To collect a useful set of images without capturing near-identical images, or many
images of light-free roads, logic was included as follows:
1. Images were captured only if the car was within `self.sim_image_grab_max_range` metres of a light,
   to avoid pictures of empty road.
2. Image capture stopped below `self.sim_image_grab_min_range` of a light, assuming it
   would be passing out of the camera frame when very nearby.
3. Another image would not be captured if the car was still within `self.sim_image_grab_min_spacing`
   metres of the point at which the last image was captured.

The car was then allowed to drive round the circuit in simulation and images were
accumulated. 285 were collected initially in `data\training_images`; this was perfectly adequate for training
the classifiers, as the simulation images were relatively easy to identify by
a DL model.

Additional simulation images were later captured as `data\training_images2`.

### Real images

All the real images were obtained from Udacity .bag files.

The first images were obtained by automatic saving in `tl_detector.py` from 
`traffic_light_training.bag` (linked to in the start project repo `README.md` file).
However, these images were of poor quality, with excessive brightness and poor
colours. Even as a human it was difficult to distinguish the colour; in some cases
it could only be determined by looking at the reflection on the car bonnet (hood).
Also, the difficulty in training a classifier on these poor images meant that we
needed more pictures.

In the second step, we used the `rviz` ROS package. The following tutorial from ROS webpage
(http://wiki.ros.org/rosbag/Tutorials/Exporting%20image%20and%20video%20data)
describes how to create a new `.launch` file and automatically capture images. Furthermore
the file `just_traffic_light.bag` linked to on the project submission page of the
classroom was used, which proved to have better-quality images.

## Image classifier and results: Faster R-CNN

### Tensorflow object detection API
One way to build a classifier is to use a powerfull API from Tensorflow on Object Detection. It has many pre-trained networks that can be fine tuned using custom datasets. There are several good tutorials that cover main steps and were used by our team as references in this project:
  - by Daniel Stang: [link](https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e)
  - by Vatsal Sodhal [link](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)
  - by Dat Tran: [link](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

As a pre-trained classifier, we chose the Faster R-CNN network ([link](https://arxiv.org/pdf/1504.08083.pdf)) that incorporates a ResNet 101 pretrained model. It provides a good balance between speed and detection accuracy for small objects on the image. In particular, we did not choose SSD (Singe Shot Detector) network as it resizes any input image to 300x300 pixels and, therefore, the accuracy of detecting small objects is reduced. The summary of main speed characteristics for different object detectors can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

### Model setup and training
Once the data was ready the following steps were taken to get classifier working:
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

### Issues:

- Back compatibility with Tensorflow 1.3 as required by Udacity. The current version of object detection API is based on 1.9 tensorflow and not compatible with 1.3. Instead we used an older version from this [commit](https://github.com/tensorflow/models/tree/d1173bc9714b5729b8c95d8e91e8647c66acebe6).

### Results

The following picures demonstrate traffic lights classifier performance for two different classes of images:

- real images taken from `just_traffic_light.bag`:
  <p float="left">
      <img src="/writeup_files/pictures/image_real_green.jpg" width="270" />
      <img src="/writeup_files/pictures/image_real_yellow.jpg" width="270" /> 
      <img src="/writeup_files/pictures/image_real_red.jpg" width="270" />
  </p>
- simulator images:
  <p float="left">
      <img src="/writeup_files/pictures/image_sim_green.jpg" width="270" />
      <img src="/writeup_files/pictures/image_sim_yellow.jpg" width="270" /> 
      <img src="/writeup_files/pictures/image_sim_red.jpg" width="270" />
  </p>
Note that classifier score is always above 90.

The videos below show 

- classifier performance in the simulator (by Udacity):
  
  [![Simulator](https://img.youtube.com/vi/n33BJwhKeUU/0.jpg)](https://youtu.be/n33BJwhKeUU)
  
- classifier performance for the `just_traffic_light.bag` file (by Udacity):
  
  [![Real images](https://img.youtube.com/vi/I5Ab-Io5ETI/0.jpg)](https://youtu.be/I5Ab-Io5ETI)


### Hardware

Tests were conducted using Nvidia GTX1070 8GB, i7-7700HQ.

## Image classifier and results: VGG

[CW: will write something here about my adventures with VGG, and include video]

## Other sections I've forgotten about

## Summary

[CW: discuss possible imporvements that could be made]
