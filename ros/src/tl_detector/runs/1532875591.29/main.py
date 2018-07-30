############################################################################### 
#   Udacity self-driving car course : Capstone Project.
#
#   Team   : smart-carla
#   Author : Charlie Wartnaby, Applus IDIADA
#   Email  : charlie.wartnaby@idiada.com
#
#   This module creates and trains a CNN to classify images containing
#   traffic lights by the colour of those lights.
#   The model setup is reused in inference mode at run time by the ROS build.
#
#   Note: adapted from semantic segregation project
############################################################################### 

import os.path
import tensorflow as tf
import time
import datetime
import warnings
from distutils.version import LooseVersion
import sys

import helper
import cnn_classifier_model

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def train_nn(sess, epochs, batch_size, get_batches_fn, cnn_model):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # DONE: Implement function

    keep_prob_value = 0.5 # After experimentation this high rate eventually does better
    learning_rate_value = 0.001 # From experiments

    # Walkthrough video help from 19:30
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        batches_run = 0
        for image, label in get_batches_fn(batch_size):
            # A note to self on sizing: Tensorflow does seem to handle the last batch being
            # smaller in size than the others, and so we can feed less data to a placeholder than
            # its allocated size, and get a smaller array out. E.g.
            # image.shape= (12, 160, 576, 3)   for a batch of 12 images x height x width x colour channels
            # but with 289 samples, the last one is:
            # image.shape= (1, 160, 576, 3)
            # and at output, we get correspondingly different sized logits tensor. 

            # Construct feed dictionary
            feed_dict = {'image_input:0'   : image,
                         'correct_label:0' : label,
                         'keep_prob:0'     : [keep_prob_value],
                         'learning_rate:0' : learning_rate_value,
                         };

            # Then actually run optimizer and get loss (OK to do in one step? Seems to work OK.)
            train_out, loss = sess.run([cnn_model.train_op, cnn_model.cross_entropy_loss], feed_dict=feed_dict)

            batches_run += 1
            total_loss += loss
            # Show progress through batches
            #sys.stdout.write('.')
            #sys.stdout.flush() 

        elapsed_time = str(datetime.timedelta(seconds=(time.time() - start_time)))
        print("")
        loss_per_batch = -1.0 if batches_run == 0 else total_loss / batches_run # avoid div by zero
        print("Epoch:", epoch, "Loss/batch:", loss_per_batch, "time so far:", elapsed_time)

    print("")

 
def run():
    proportion_train = 0.8 # rest validation. Don't have big enough set for separate test set really!
    img_type = "both"   # "sim", "real" or "both"
    save_model_name = "both_full_frame_model.ckpt" # if saving this time

    load_trained_weights = False   # False to train, True to run in inference mode
    
    if load_trained_weights:
        # Want to apply model in inference mode to all images
        proportion_train = 0.0

    runs_dir = "../ros/src/tl_detector/runs"

    # Walkthrough: maybe ~6 epochs to start with. Batches not too big because large amount of information.
    epochs = 20 # Now reduced for debug   8 # Seemed to get worse last time after too many
    batch_size = 1 # Already getting memory warnings!
    # Other hyperparameters in train_nn(); would have put them here but went with template calling structure


    # Split images into training and validation sets
    training_dir_list=['../data/training_images',
                       '../data/training_images3/green',
                       '../data/training_images3/yellow',
                       '../data/training_images3/red',
                       '../data/training_images3/unknown']
    training_image_paths, validation_image_paths =  \
                helper.get_split_image_paths(proportion_train, img_type, training_dir_list)

    with tf.Session() as sess:

        # Construct model; done by separate class so that it can be invoked
        # by ROS runtime in exactly the same way
        cnn_model = cnn_classifier_model.CnnClassifierModel(sess, load_trained_weights)
        
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(training_image_paths, cnn_model.image_shape, cnn_model.num_classes)

        if not load_trained_weights:
            print("Training model...")
            train_nn(sess, epochs, batch_size, get_batches_fn, cnn_model)
        else:
            # Using trained weights so no training to do
            pass

        # DONE: Save inference data using helper.save_inference_samples
        run_output_dir = helper.save_inference_samples(runs_dir, validation_image_paths, sess, 
                   cnn_model.image_shape, cnn_model.keep_prob, cnn_model.logits, cnn_model.image_input)

        # Save model for reuse in inference mode, if we trained it this time
        if not load_trained_weights:
            save_path = cnn_model.saver.save(sess, os.path.join(run_output_dir, save_model_name))
            print("Saved TensorFlow model in %s\n" % save_path)
        else:
            print("Didn't save model because we loaded the weights from disk this time")

if __name__ == '__main__':
    run()
