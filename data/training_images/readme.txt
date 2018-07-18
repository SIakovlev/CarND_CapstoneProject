The 'real' images are taken from the ROS .bag file linked to in
the starter repo for this project. They are photos taken from the vehicle
in the Udacity car park, mostly looking at a single temporary traffic light.
The 'sim' images are exported from the /image_color topic as received from the
Unity simulator.

The filename convention is:

[real|sim]_index_category.jpg

where index is an arbitrary integer, and category is the same as the
light state message enumeration, i.e.:

0=RED
1=YELLOW (AMBER)
2=GREEN
4=UNKNOWN (is that right?)
