#!/bin/bash
for((i=872;i<1800;i+=1));do mkdir data/vision_reach/color_data/color_$i; for((j=0;j<12;j+=1));do python pybullet_reacher_example.py --gif_file=data/vision_reach/color_data/color_$i/example_$j.gif --object_type=circle --object_scale=40 --task_color_index=$i --image_height=128 --image_width=128 --num_frames=24; done; done
