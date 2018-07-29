import os
import os.path
import six.moves.urllib as urllib
import time, sys

from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

DETECTION_THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self):

        self.is_site = True # if set then classifier for real images will be used

        # specify path to /models directory with respect to the absolute path of tl_classifier.py
        MODEL_DIR_NAME=os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(MODEL_DIR_NAME):
            os.makedirs(MODEL_DIR_NAME)

        # specify the model name based on the is_site flag state
        if self.is_site:    
            FILENAME = 'faster_rcnn_real.pb'
            # Dropbox link for downloading (backup if web location is down)
            #DOWNLOAD_URL='https://www.dropbox.com/s/e3acw9s1wtvzk3n/faster_rcnn_real.pb?dl=1'
            DOWNLOAD_URL='http://www.wartnaby.org/smart_carla/faster_rcnn_real.pb'
        else:
            FILENAME = 'faster_rcnn_sim.pb'
            # Dropbox link for downloading (backup if web location is down)
            #DOWNLOAD_URL='https://www.dropbox.com/s/wc4bky8v7roya0q/faster_rcnn_sim.pb?dl=1'
            DOWNLOAD_URL='http://www.wartnaby.org/smart_carla/faster_rcnn_sim.pb'

        # full path to the model file
        fullfilename = os.path.join(MODEL_DIR_NAME, FILENAME)

        # Function that reports downloading status
        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                            (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        if os.path.isfile(fullfilename):
            print("Model file is downloaded. Full path: {} \n".format(fullfilename))
            print("Proceeding with graph initialisation...\n")
        else:
            print("Model file is missing, start downloading...\n")
            urllib.request.urlretrieve(DOWNLOAD_URL, fullfilename, reporthook)
            print()
            print("New directory was created: {}".format(MODEL_DIR_NAME))
            print("Model file is downloaded. Full path: {} \n".format(fullfilename))
            print("Proceeding with classifier initialisation...\n")

        # Import tensorflow graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(fullfilename, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            # get all necessary tensors
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # BGR to RGB conversion
            img = image[:, :, ::-1]
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            # run classifier
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            # find the top score for a given image frame
            top_score = np.amax(np.squeeze(scores))
            # figure out traffic light class based on the top score
            if top_score > DETECTION_THRESHOLD:
                tl_state = int(np.squeeze(classes)[0])
                if tl_state == 1:
                    sys.stderr.write("Debug: Traffic state: RED, score=%.2f\n" % (top_score*100))
                    return TrafficLight.RED
                elif tl_state == 2:
                    sys.stderr.write("Debug: Traffic state: YELLOW, score=%.2f\n" % (top_score*100))
                    return TrafficLight.YELLOW
                else:
                    sys.stderr.write("Debug: Traffic state: GREEN, score=%.2f\n" % (top_score*100))
                    return TrafficLight.GREEN
            else:
                sys.stderr.write("Debug: Traffic state: OFF\n")     
                return TrafficLight.UNKNOWN

        '''
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
        '''
        
