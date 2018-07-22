import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def get_split_image_paths(proportion_train, img_type, data_folder):
    """Return file paths for images to use for training, and validation"""

    # Select simulation, real or both types of image
    if img_type == "both":
        wildcard = "*.jpg"
    else:
        # "sim" or "real"
        wildcard = img_type + "_*.jpg"

    image_paths = glob(os.path.join(data_folder, wildcard))
    random.shuffle(image_paths)
    num_train = int(round(len(image_paths) * proportion_train))

    print("Selected %d total images (%d for training, propn=%f)" 
             % (len(image_paths), num_train, proportion_train))

    return image_paths[:num_train], image_paths[num_train:]

def get_numeric_light_state_from_filename(image_file):
    """Gets 0,1,2,4 integer from end of filename and returns 0-3 integer accordingly"""

    filename_with_ext = os.path.basename(image_file)          # e.g. "real_1980_4.jpg"
    filename_wo_ext = os.path.splitext(filename_with_ext)[0]  # e.g. "real_1980_4"
    filename_parts = filename_wo_ext.split('_')               # e.g. ["real", "1980", 4"]
    if len(filename_parts) != 3:
        sys.stderr.write("Error: image filename not like real_1980_4: \n" % image_file)
        sys.exit(1)
    numeric_state = int(filename_parts[2]) # or ValueException I guess if not integer
    if numeric_state == 4:
        numeric_state = 3 # so we have consecutive range 0..3 for red, yellow, green, unknown

    return numeric_state

def gen_batch_function(image_paths, image_shape, num_classes):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    # CW: generator function means only one batch at a time loaded into memory, yields
    #     control to caller on each iteration
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """

        for batch_i in range(0, len(image_paths), batch_size): # divide full (shuffled) set into batches
            images = []
            true_classes_onehot = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                numeric_state = get_numeric_light_state_from_filename(image_file)
                onehot = np.zeros(num_classes)
                onehot[numeric_state] = 1

                images.append(image)                # so now have 4D for data, i.e. [image, height, width, colour channels]
                true_classes_onehot.append(onehot)  # so now have 2D for ground truth, i.e. [image, classes]

            yield np.array(images), np.array(true_classes_onehot) # return this batch
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, image_paths, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :param image_paths: File paths of images to test
    :return: Output for for each test image
    """
    num_correct = 0
    for image_file in image_paths:
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        #print("Debug im_softmax: " + repr(im_softmax))
        # Figure out if we got it right
        top_score = np.argmax(im_softmax)
        correct_state = get_numeric_light_state_from_filename(image_file)
        prediction_correct = (top_score == correct_state)
        num_correct += 1 if prediction_correct else 0

        # Embed scores for different colours in filename to help understand output
        softmax_scores_as_list = im_softmax[0].tolist()[0]
        numeric_state = get_numeric_light_state_from_filename(image_file)
        classification = "_R%.2f_Y%.2f_G%.2f_U%.2f" % tuple(softmax_scores_as_list)
        # Debug -- is there some problem which means we're getting _identical_ logits each time?
        #print("Debug softmax_scores_as_list: " + repr(softmax_scores_as_list))
        basename = os.path.basename(image_file)
        root, ext = os.path.splitext(basename)
        newname = "P_" if prediction_correct else "F_" # pass or fail
        newname += root + classification + ext # add detailed scores to filename

        yield newname, image # just return original image so we can look at it with new filename

    print("Test set complete, %d of %d correct (proportion=%f)\n" %
             (num_correct, len(image_paths), num_correct/len(image_paths)))

def save_inference_samples(runs_dir, image_paths, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, image_paths, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    return output_dir