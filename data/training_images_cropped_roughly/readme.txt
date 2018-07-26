The 'real' images are taken from the ROS .bag file linked to in
the starter repo for this project. They are photos taken from the vehicle
in the Udacity car park, mostly looking at a single temporary traffic light.
The 'sim' images are exported from the /image_color topic as received from the
Unity simulator.

In this directory, the original full-frame images have been cropped around
a candidate traffic light, as if an object detector like SSD or R-CNN
had identified a bounding box. This is made deliberately a bit rough so that
the network is robust to some noise in where the bounding boxes are drawn,
sometimes including some background, sometimes cutting into light itself
a little, etc. The cropping was done by hand. As some of the simulator images
showed the *back* of some traffic lights, some images of those were included
with a category of 'unknown', too.

I usually just selected one of the lights of the 2 or 3 visible in each
raw image, but resaved some of the images with a yellow light under a new
name so that I could use more than one light, as we don't have that many
yellow samples.

Currently gave up after sim_1705_2.jpg, but could copy over sim_1706+ files
from ../training_images and crop those similarly if the dataset seems too small.

The filename convention is:

[real|sim]_index_category.jpg

where index is an arbitrary integer, and category is the same as the
light state message enumeration, i.e.:

0=RED
1=YELLOW (AMBER)
2=GREEN
4=UNKNOWN (is that right?)

The category was appended automatically when running with the simulator, but
done by hand for the photos where the ground truth state was not provided.
