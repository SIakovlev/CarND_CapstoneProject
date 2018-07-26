############################################################################### 
#    Udacity self-driving car course : Semantic Segregation Project.
#
#   Author : Charlie Wartnaby, Applus IDIADA
#   Email  : charlie.wartnaby@idiada.com
#
############################################################################### 

import os.path
import tensorflow as tf
import helper
import time
import datetime
import warnings
from distutils.version import LooseVersion


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # DONE: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name      = 'image_input:0' # Walkthrough: so we can pass through image
    vgg_keep_prob_tensor_name  = 'keep_prob:0'   # Walkthrough: so we can adjust fraction of data retained to avoid overfitting, though weights not frozen (says are in walkthrough but note corrects that)
    vgg_layer3_out_tensor_name = 'layer3_out:0'  # Walkthrough: pool3 layer as shown in paper architecture
    vgg_layer4_out_tensor_name = 'layer4_out:0'  # Walkthrough: pool4 layer
    vgg_layer7_out_tensor_name = 'layer7_out:0'  # Walkthrough: pool5 layer
    
    # Following walkthrough tips
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return (image_input, keep_prob, layer3_out, layer4_out, layer7_out)



def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # DONE: Implement function

    # See also lesson "FCN-8 Decoder" for structure, and Long_Shelhamer paper

    # Walkthrough video started with 1x1 convolution like this, but notes explained
    # that was already done for us (loaded model is not ordinary VGG but already
    # adapted for FCN). In fact the VGG network provided looks very much like
    # the one generated by the Single-Shot Detector caffe code, so I guess they
    # share some common heritage.
    #conv_1x1 = tf.layers.conv2d(vgg_layer7_out, # at/near end of VGG
    #                            num_classes, # just road/nonroad for us
    #                          1, # as 1x1 conv
    #                          padding='same',
    #                          kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # Using Tensorboard to visualise the structure of the VGG model provided, and
    # tf.trainable_variables() to list the dimensions and sizes of the weights and biases
    # for each layer, I arrive at this summary of what shape the output of each layer
    # is (knowing that we started with a 160 height x 576 width x 3 colour channel image).
    # All of the convolution layers have SAME padding and [1,1,1,1] strides so they
    # don't reduce the x-y pixel size. All the pooling layers have [1,2,2,1] strides so
    # they halve the pixel size. I'm ignoring the first dimension (across images), as
    # everything works on one image at a time.
    #
    # Layer name  Details                     Output dimensions
    # <input>     raw image                   288x384x3
    # conv1_1     conv2d 3x3x3x64, Relu       288x384x64
    # conv1_2     conv2d 3x3x64x64, Relu      288x384x64
    # pool1       pool [1,2,2,1]              144x192x64
    # conv2_1     conv2d 3x3x64x128, Relu     144x192x128
    # conv2_2     conv2d 3x3x128x128, Relu    144x192x128
    # pool2       pool [1,2,2,1]              72x96x128
    # conv3_1     conv2d 3x3x128x256, Relu    72x96x256
    # conv3_2     conv2d 3x3x256x256, Relu    72x96x256
    # conv3_3     conv2d 3x3x256x256, Relu    72x96x256
    # pool3       pool [1,2,2,1]              36x48x256     --> layer3_out
    # conv4_1     conv2d 3x3x256x512, Relu    36x48x512
    # conv4_2     conv2d 3x3x512x512, Relu    36x48x512
    # conv4_3     conv2d 3x3x512x512, Relu    36x48x512
    # pool4       pool [1,2,2,1]              18x24x512     --> layer4_out
    # conv5_1     conv2d 3x3x512x512, Relu    18x24x512
    # conv5_2     conv2d 3x3x512x512, Relu    18x24x512
    # conv5_3     conv2d 3x3x512x512, Relu    18x24x512
    # pool5       pool [1,2,2,1]              9x12x512
    # fc6         conv2d 7x7x512x4096, Relu   9x12x4096
    # dropout     dropout(keep_prob)          9x12x4096
    # fc7         conv2d 1x1x4096x4096, Relu  9x12x4096
    # dropout_1   dropout(keep_prob)          9x12x4096     --> layer7_out

    # To get something working just go straight from final layer to fully
    # connected with depth equal to number of classes we require
    # Borowed from https://www.tensorflow.org/versions/r1.0/tutorials/layers
    # _cw suffixes on layer names to avoid inadvertent links to VGG names!
    flat_cw = tf.reshape(vgg_layer7_out, [-1,9*12*4096], name="flat_cw")
    dense_cw = tf.layers.dense(inputs=flat_cw, units=32, activation=tf.nn.relu, name="dense_cw") # CW far fewer than example
    dropout_cw = tf.layers.dropout(inputs=dense_cw, rate=0.4, name="dropout_cw")
    final_layer_cw = tf.layers.dense(inputs=dropout_cw, units=num_classes, name="final_layer_cw")

    return final_layer_cw # should be num images x num_classes



def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # DONE: Implement function

    # Walkthrough video help from 17:30

    # See also lesson FCN-8 - Classification & Loss

    # have to reshape tensor to 2D to get logits.
    # Naming tensors to make debug easier if necessary
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    
    # Reshape labels before feeding to TensorFlow session

    # Similar code to traffic sign classifier project now:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label, name='cross_entropy')
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    train_op = optimizer.minimize(cross_entropy_loss, name='train_op')

    return (logits, train_op, cross_entropy_loss)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate):
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
            # Labels are 4D [N image, height, width, classes]; we just want
            # to span pixels overall and classes for comparison with the network output

            # A note to self on sizing: Tensorflow does seem to handle the last batch being
            # smaller in size than the others, and so we can feed less data to a placeholder than
            # its allocated size, and get a smaller array out. E.g.
            # image.shape= (12, 160, 576, 3)   for a batch of 12 images x height x width x colour channels
            # but with 289 samples, the last one is:
            # image.shape= (1, 160, 576, 3)
            # and at output, we get corresponding logits_out.shape= (1105920, 2) and logits_out.shape= (92160, 2)
            # respectively, where 12*160*576=1105920 and 1*160*576=92160.

            # Construct feed dictionary
            feed_dict = {'image_input:0'   : image,
                         'correct_label:0' : label,
                         'keep_prob:0'     : [keep_prob_value],
                         'learning_rate:0' : learning_rate_value,
                         };

            # Then actually run optimizer and get loss (OK to do in one step? Seems to work OK.)
            train_out, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            batches_run += 1
            total_loss += loss
            print('.', end='', flush=True) # Show progress through batches

        elapsed_time = str(datetime.timedelta(seconds=(time.time() - start_time)))
        print("")
        loss_per_batch = -1.0 if batches_run == 0 else total_loss / batches_run # avoid div by zero
        print("Epoch:", epoch, "Loss/batch:", loss_per_batch, "time so far:", elapsed_time, end='', flush=True)

    print("")

 
def run():
    num_classes = 4 #  CW: red, yellow, green, unknown
    proportion_train = 0.75 # rest validation. Don't have big enough set for separate test set really!
    img_type = "sim"   # "sim", "real" or "both"

    # CW: both real Carla images and simulator exports are 800x600.
    # We might find shrinking them helps with performance in terms of
    # speed or memory, though classification quality will suffer if 
    # we go too far. Semantic segregation project chose a size with
    # reasonably high power-of-two factors to allow for the repeated halving
    # of resolution going up the CNN funnel (160x576, or 2^5*5 x 2^6*9)
    # without any awkward padding issues. 800 already divides nicely,
    # but 600 is 2^3*3*5^2 so it can only be halved cleanly 3 times.
    # But there is not too much happening at the bottom of any of our
    # images, so clipping a little to 800x576 should be quite nice,
    # maybe with a 1/2 or 1/4 shrink to speed things up.
    # TODO clipping logic -- for now just shrinking to avoid code changes
    image_shape = (288, 384) # Initial experiment size (heightxwidth) -- out of GPU memory trying 576*800. Multiples of 32.

    data_dir = './data'
    runs_dir = './runs'

    # Walkthrough: maybe ~6 epochs to start with. Batches not too big because large amount of information.
    epochs = 20 # To get started
    batch_size = 1 # Already getting memory warnings!
    # Other hyperparameters in train_nn(); would have put them here but went with template calling structure

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Split images into training and validation sets
        training_image_paths, validation_image_paths =  \
                    helper.get_split_image_paths(proportion_train, img_type, '../data/training_images')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(training_image_paths, image_shape, num_classes)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Walkthrough: correct labels will be 4D (batch, height, width, num classes)
        # CW: see my comments in get_batches_fn() to remind self of why... final (num classes) axis is one-hot
        #     with [0]=1 for background and [1]=1 for (any) road

        # DONE: Build NN using load_vgg, layers, and optimize function

        # CW: load VGG16 (actually already modified version for FCN) and pick out tensors corresponding
        #     to layers we want to attach to
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

        # CW: add our own layers to do transpose convolution skip connections from encoder
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes) # get final layer out

        # CW: for debug, want to visualise model structure in Tensorboard; initially did this
        # before adding my layers to understand how to connect to unmodified VGG layers. Now
        # doing afterwards to include picture in write-up that includes my layers.
        if True:  # Turned off for most runs when not debugging
            print(tf.trainable_variables()) # also trying to understand what we've got
            log_path = os.path.join(vgg_path, 'logs')
            writer = tf.summary.FileWriter(log_path, graph=sess.graph)
            # Then visualise as follows:
            # >tensorboard --logdir=C:\Users\UK000044\git\CarND-Semantic-Segmentation\data\vgg\logs --host localhost
            # Open http://localhost:6006 in browser (if don't specify --host, in Windows 10 uses PC name, and 
            #                         localhost or 127.0.0.1 find no server, whereas http://pc_name:6006 does work)

        # CW: add operations to classify each pixel by class and assess performance
        # Input label size dynamic because have odd number of images as last batch; can get away without specifying 
        # shape in complete detail up front but specifying those we know to hopefully make bugs more apparent
        correct_label = tf.placeholder(tf.float32, shape=[None,num_classes], name='correct_label')

        # Reshape labels as one-hot matrix spanning all of the pixels from all of the images concatenated together
        flattened_label = tf.reshape(correct_label, (-1, num_classes), name='flattened_label')

        learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # CW: have to initialise variables at some point
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # DONE: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 flattened_label, keep_prob, learning_rate)

        # DONE: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, validation_image_paths, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
