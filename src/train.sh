#!/bin/bash

python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 5 --obstacles_input_type dynamic --max_epochs 10 --record False
python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 6 --obstacles_input_type dynamic_relvel --max_epochs 10 --record False
python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 7 --obstacles_input_type dynamic_radii --record False
python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 8 --obstacles_input_type dynamic_radii_relvel --max_epochs 10 --record False
python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 9 --obstacles_input_type dynamic_points6 --max_epochs 10 --record False
python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 10 --obstacles_input_type dynamic_points6_relvel --max_epochs 10 --record False

# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 15 --obstacles_input_type dynamic --max_epochs 1
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 16 --obstacles_input_type dynamic_relvel --max_epochs 1 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 17 --obstacles_input_type dynamic_radii --max_epochs 1 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 18 --obstacles_input_type dynamic_radii_relvel --max_epochs 1 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 19 --obstacles_input_type dynamic_points6 --max_epochs 1 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 20 --obstacles_input_type dynamic_points6_relvel --max_epochs 1 --record False

# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 11 --obstacles_input_type dynamic --max_epochs 1

notify-send "Script has finished running"