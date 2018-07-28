from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import scipy
import sys

sys.path.insert(0,"../../../training")
import cnn_classifier_model


class TLClassifier(object):
    def __init__(self):
        self.sess = tf.Session()
        self.cnn_model = cnn_classifier_model.CnnClassifierModel(self.sess, True)
        self.image_counter = 0

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Shrink image to size required by network first, so that any further
        # operations are computationally cheaper
        image = scipy.misc.imresize(image, self.cnn_model.image_shape)

        # Convert CV2 BGR encoding to RGB as used to train model
        image = image[...,::-1]

        # Actually run image through network in inference mode to get classification
        im_softmax = self.sess.run(
            [tf.nn.softmax(self.cnn_model.logits)],
            {self.cnn_model.keep_prob: 1.0, self.cnn_model.image_input: [image]})

        # Identify most likely class (light colour)
        top_score = np.argmax(im_softmax)
        if (top_score == 3):
            # Convert 0..3 model output to ROS light state value
            top_score = 4 # unknown

        if True: # debug only
            # Useful to see scores for each class. Also saving images has been useful
            # to identify colour coding problems and so that we can see what the image
            # looked like on occasions when it was misclassified
            softmax_scores_as_list = im_softmax[0].tolist()[0]
            classification = "R%.2f_Y%.2f_G%.2f_U%.2f" % tuple(softmax_scores_as_list)
            filename = "run_%d_%d_%s.jpg" % (self.image_counter, top_score, classification)
            #scipy.misc.imsave(filename, image)
            print("Debug: image %s classed:%d" % (filename, top_score))
            self.image_counter += 1
            
             
        return top_score
        
