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

        # specify path to /models directory with respect to the absolute path of tl_classifier.py
        MODEL_DIR_NAME=os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(MODEL_DIR_NAME):
            os.makedirs(MODEL_DIR_NAME)
        # specify the model name
        FILENAME = 'faster_fcnn.pb'
        # Dropbox link for downloading
        DOWNLOAD_URL='https://www.dropbox.com/s/9yuj9ve5a36dguo/faster_fcnn.pb?dl=1'
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
        
