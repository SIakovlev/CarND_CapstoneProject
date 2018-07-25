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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = scipy.misc.imresize(image, self.cnn_model.image_shape)

        im_softmax = self.sess.run(
            [tf.nn.softmax(self.cnn_model.logits)],
            {self.cnn_model.keep_prob: 1.0, self.cnn_model.image_input: [image]})

        # Figure out if we got it right
        top_score = np.argmax(im_softmax)
        if (top_score == 3):
            top_score = 4 # unknown

        print("Debug: image classification=%d" % top_score)
             
        return top_score
        
