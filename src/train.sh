#!/bin/bash

# python3 master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 5 --obstacles_input_type dynamic
# python3 master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 6 --obstacles_input_type dynamic_relvel
python3 master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 7 --obstacles_input_type dynamic_radii
python3 master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 8 --obstacles_input_type dynamic_radii_relvel
python3 master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 9 --obstacles_input_type dynamic_points6
python3 master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 10 --obstacles_input_type dynamic_points6_relvel