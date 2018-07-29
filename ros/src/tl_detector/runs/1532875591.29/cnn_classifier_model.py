############################################################################### 
#   Udacity self-driving car course : Capstone Project.
#
#   Team   : smart-carla
#   Author : Charlie Wartnaby, Applus IDIADA
#   Email  : charlie.wartnaby@idiada.com
#
#   This module has the common model setup tasks used both in
#   training a CNN, and setting up the same model to use at
#   run time in inference mode. It was originally derived
#   from the semantic segregation project.
#
############################################################################### 

import os.path
import tensorflow as tf

import helper

class CnnClassifierModel():
    
    def load_vgg(self, sess, vgg_path):
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

        self.image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        self.keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        self.layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        self.layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        self.layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)


    def add_custom_layers(self):
        """
        Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
        :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
        :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
        :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
        :param num_classes: Number of classes to classify
        :param image_shape: tuple of rows, columns dimensions of each image feeding network
        :return: The Tensor for the last layer of output
        """
        # See also lesson "FCN-8 Decoder" for structure, and Long_Shelhamer paper

        # Using Tensorboard to visualise the structure of the VGG model provided, and
        # tf.trainable_variables() to list the dimensions and sizes of the weights and biases
        # for each layer, I arrive at this summary of what shape the output of each layer
        # is (knowing that we started with a 288 height x 384 width x 3 colour channel image).
        # All of the convolution layers have SAME padding and [1,1,1,1] strides so they
        # don't reduce the x-y pixel size. All the pooling layers have [1,2,2,1] strides so
        # they halve the pixel size. I'm ignoring the first dimension (across images), as
        # everything works on one image at a time.
        #
        # Layer name  Details                     Output dimensions [example, h x w may vary]
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

        # Go from final layer to fully
        # connected with depth equal to number of classes we require
        # with an intermediate fully connected layer
        # Borowed from https://www.tensorflow.org/versions/r1.0/tutorials/layers
        # _cw suffixes on layer names to avoid inadvertent links to VGG names!
        final_rows = int(self.image_shape[0] / 32)
        final_cols = int(self.image_shape[1] / 32)
        flat_cw = tf.reshape(self.layer7_out, [-1,final_rows*final_cols*4096], name="flat_cw")
        dense_cw = tf.layers.dense(inputs=flat_cw, units=128, activation=tf.nn.relu, name="dense_cw") # number of units chosen not to exceed memory! Classifier useless with fewer.
        final_layer_cw = tf.layers.dense(inputs=dense_cw, units=self.num_classes, name="final_layer_cw")

        # End-to-end training on our small dataset didn't really work; the number
        # of images was small compared to the sheer size of the VGG-based model.
        # Instead, froze the VGG weights and trained only the custom layer weights.
        # So here, we identify the variables for those custom layer weights
        # so that we can specify that the optimiser only adjusts these.
        with tf.variable_scope('dense_cw', reuse=True):
            dense_weights=tf.get_variable('kernel')
            dense_biases = tf.get_variable('bias')
        with tf.variable_scope('final_layer_cw', reuse=True):
            final_weights=tf.get_variable('kernel')
            final_biases = tf.get_variable('bias')

        opt_var_list=[dense_weights, dense_biases, final_weights, final_biases]

        # Final layer output is 2D: number of images x number of classes
        return final_layer_cw, opt_var_list



    def build_optimizer(self):
        """
        Build the TensorFLow loss and optimizer operations.
        Note: in inference mode, we don't need those, but adding them anyway so that
        the saved model weights still "fit" -- not sure if that's really required.
        
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the correct label image
        :param learning_rate: TF Placeholder for the learning rate
        :param num_classes: Number of classes to classify
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """
        # See also lesson FCN-8 - Classification & Loss

        # have to reshape tensor to 2D to get logits.
        # Naming tensors to make debug easier if necessary
        logits = tf.reshape(self.final_layer, (-1, self.num_classes), name='logits')
        
        # Reshape labels before feeding to TensorFlow session

        # Similar code to traffic sign classifier project now:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.correct_label, name='cross_entropy')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        train_op = optimizer.minimize(cross_entropy_loss, name='train_op',
                                           var_list=self.opt_var_list)

        return (logits, train_op, cross_entropy_loss)

    def __init__(self, sess, load_trained_weights):

        self.num_classes = 4 #  CW: red, yellow, green, unknown
        
        # CW: both real Carla images and simulator exports are 800x600.
        # We might find shrinking them helps with performance in terms of
        # speed or memory, though classification quality will suffer if 
        # we go too far. Semantic segregation project chose a size with
        # reasonably high power-of-two factors to allow for the repeated halving
        # of resolution going up the CNN funnel (160x576, or 2^5*5 x 2^6*9)
        # without any awkward padding issues. 800 already divides nicely,
        # but 600 is 2^3*3*5^2 so it can only be halved cleanly 3 times.
        # But there is not too much happening at the bottom of any of our
        # images, so clipping a little off the bottom should be quite nice,
        # maybe with a 1/2 or 1/4 shrink to speed things up.
        # But for now just shrinking whole image to manageable size.
        # Shape is rows x cols (i.e. height x width), not usual width x height!
        #image_shape = (2*32, 1*32) # Portrait slot shape suitable for crops round traffic lights
        self.image_shape = (9*32, 12*32) # Landscape format for full frames -- out of GPU memory trying 576*800, had to go smaller.
        # This size is bare minimum to get model to usefully classify on original dataset.

        data_dir = 'data'
        restore_from_model_path = "runs/1532796265.11/both_full_frame_model.ckpt"
        if os.getcwd().endswith("training"):
            # Horrible hack
            rel_path_to_ros_runtime = "../ros/src/tl_detector/"
            data_dir = os.path.join(rel_path_to_ros_runtime, data_dir)
            restore_from_model_path = os.path.join(rel_path_to_ros_runtime, restore_from_model_path)

        # Download pretrained vgg model
        helper.maybe_download_pretrained_vgg(data_dir)

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # CW: load VGG16 (actually already modified version for FCN) and pick out tensors corresponding
        #     to layers we want to attach to
        self.load_vgg(sess, vgg_path)

        # CW: add our own layers to do transpose convolution skip connections from encoder
        self.final_layer, self.opt_var_list = self.add_custom_layers()

        # CW: for debug, want to visualise model structure in Tensorboard; initially did this
        # before adding my layers to understand how to connect to unmodified VGG layers. Now
        # doing afterwards to include picture in write-up that includes my layers.
        if False:  # Turned off for most runs when not debugging
            print(tf.trainable_variables()) # also trying to understand what we've got
            log_path = os.path.join(vgg_path, 'logs')
            writer = tf.summary.FileWriter(log_path, graph=sess.graph)
            # Then visualise as follows:
            # >tensorboard --logdir=C:\Users\UK000044\git\CarND-Semantic-Segmentation\data\vgg\logs --host localhost
            # Open http://localhost:6006 in browser (if don't specify --host, in Windows 10 uses PC name, and 
            #                         localhost or 127.0.0.1 find no server, whereas http://pc_name:6006 does work)

        # CW: add operations to classify each image by class and assess performance
        # Input label size dynamic because have odd number of images as last batch; can get away without specifying 
        # shape in complete detail up front but specifying those we know to hopefully make bugs more apparent
        self.correct_label = tf.placeholder(tf.float32, shape=[None,self.num_classes], name='correct_label')

        # Reshape labels as one-hot matrix spanning all of the images concatenated together
        self.flattened_label = tf.reshape(self.correct_label, (-1, self.num_classes), name='flattened_label')

        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

        # Build optimiser (doesn't actually get run as yet)
        self.logits, self.train_op, self.cross_entropy_loss = self.build_optimizer()

        # Add operations to save and restore the variables
        self.saver = tf.train.Saver()

        # CW: have to initialise variables at some point
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        if load_trained_weights:
            # Pretrained weights are stored in a large file, too big for GitHub,
            # so download those now if we haven't already got them
            print("Loading model weights from %s" % restore_from_model_path)
            url_folder = "http://www.wartnaby.org/smart_carla/training/runs/1532796265.11/"
            local_folder = os.path.dirname(restore_from_model_path)
            file_list = ["both_full_frame_model.ckpt.data-00000-of-00001",
                         "both_full_frame_model.ckpt.index",
                         "both_full_frame_model.ckpt.meta",
                         "checkpoint"]
            helper.maybe_download_files_of_same_name_from_server(url_folder, local_folder, file_list)
            self.saver.restore(sess, restore_from_model_path)    

        print("CNN classifier model constructed")

