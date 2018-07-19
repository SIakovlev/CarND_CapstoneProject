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


def gen_batch_function(data_folder, image_shape):
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
        # CW: Kitti dataset. Each photo has two label images.
        #  data/data_road/image/training/image_2 = collection of .png colour photos of roads e.g. umm_000042.png
        #  data/data_road/image/training/gt_image_2 = .png of ground truth (gt) where magenta=our lane/road,
        #                              black=other road, red=something else
        #      e.g. umm_road_000042.png for ground truth whole road, umm_lane_000042.png = just our lane
        #  prefixes um_, umm_, uu_: http://www.cvlibs.net/datasets/kitti/eval_road.php
        #       uu - urban unmarked (98/100)
        #       um - urban marked (95/96)
        #       umm - urban multiple marked lanes (96/94)

        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png')) # so raw photos

        # CW: Make dictionary to look up road (not lane) ground truth image for each photo
        #     e.g. umm_000042.png -> umm_road_000042.png
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        background_color = np.array([255, 0, 0]) # CW: red

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size): # divide full (shuffled) set into batches
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)       # real photo
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape) # ground truth image

                gt_bg = np.all(gt_image == background_color, axis=2) # CW: for each pixel, is it background (red)?
                
                # CW: for each pixel now in 2D array, adding a further array dimension spanning classes -- but
                #     only have one class, for road/not road, so has size 1 to start with:
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1) # unpacks existing shape tuple as positional arguments -- adding a dimension of size 1?
                # ... so now have 3 dimensions for each image (height, width, classes)

                # ... then add element to each class dimension for opposite
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2) # inverted so 1=non-background
                # CW: so for each point, now have one-hot array like [0,1] (not background, is road) or
                #                                                    [1,0] (is background, not road)

                # TODO -- so network will identify 'other' roads, (black in ground truth images),
                #         not just 'our' road (magenta in images) -- is that OK/intended?

                images.append(image)
                gt_images.append(gt_image)  # so now have 4D for ground truth, i.e. [image, height, width, classes] (or w,h, not sure)

            yield np.array(images), np.array(gt_images) # return this batch
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
