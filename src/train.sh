#!/bin/bash

# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 5 --obstacles_input_type dynamic --max_epochs 10 --record False --train False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 6 --obstacles_input_type dynamic_relvel --max_epochs 10 --record False --train False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 7 --obstacles_input_type dynamic_radii --record False --train False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 8 --obstacles_input_type dynamic_radii_relvel --max_epochs 10 --record False --train False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 9 --obstacles_input_type dynamic_points6 --max_epochs 10 --record False --train False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 10 --obstacles_input_type dynamic_points6_relvel --max_epochs 10 --record False --train False

# python3 src/master_script_custom.py --model_name varyingNQuadsRNN_v2 --model_number 11 --obstacles_input_type none --max_epochs 10 --record False --train False

# python3 src/master_script_custom.py --model_name varyingNQuadsRNN_v2 --model_number 12 --others_input_type relpos_relvel --obstacles_input_type none --max_epochs 10 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 13 --others_input_type relpos_relvel --obstacles_input_type dynamic --max_epochs 10 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 14 --others_input_type relpos_relvel --obstacles_input_type dynamic_relvel --max_epochs 10 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 15 --others_input_type relpos_relvel --obstacles_input_type dynamic_radii --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 16 --others_input_type relpos_relvel --obstacles_input_type dynamic_radii_relvel --max_epochs 10 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 17 --others_input_type relpos_relvel --obstacles_input_type dynamic_points6 --max_epochs 10 --record False
# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 18 --others_input_type relpos_relvel --obstacles_input_type dynamic_points6_relvel --max_epochs 10 --record False

# python3 src/master_script_custom.py --model_name varyingNQuadsRNN_v2 --model_number 17 --warmstart True --max_epochs 10

# python3 src/master_script_custom.py --model_name onlyEllipsoidObstaclesRNN --model_number 18 \
# --others_input_type none --obstacles_input_type dynamic \
# --datasets_training "dynObs10quad1_1 dynObs10quad1_2 dynObs10quad1_3" --datasets_validation "dynObs10quad1_4" --datasets_testing "dynObs10quad1_4"

# python3 src/master_script_custom.py --model_name dynamicEllipsoidObstaclesRNN --model_number 19 \
# --others_input_type relpos_relvel --obstacles_input_type dynamic \
# --datasets_training "dynObs10quad10_1 dynObs10quad10_5 dynObs10quad10_6 dynObs10quad10_bugged dynObs10quad10_2" --datasets_validation "dynObs10quad10_3 dynObs10quad10_4" --datasets_testing "dynObs10quad10_3 dynObs10quad10_4"

# python3 src/master_script_custom.py --model_name onlyEllipsoidObstaclesRNN --model_number 20 \
# --others_input_type none --obstacles_input_type dynamic \
# --datasets_training "dynObs10quad1_1 dynObs10quad1_2 dynObs10quad1_3" --datasets_validation "dynObs10quad1_4" --datasets_testing "dynObs10quad1_4"

# python3 src/master_script_custom.py --model_name onlyEllipsoidObstaclesRNN --model_number 21 \
# --others_input_type none --obstacles_input_type dynamic_relvel \
# --datasets_training "dynObs10quad1_1 dynObs10quad1_2 dynObs10quad1_3" --datasets_validation "dynObs10quad1_4" --datasets_testing "dynObs10quad1_4"

python3 src/master_script_custom.py --model_name onlyEllipsoidObstaclesRNN --model_number 22 \
--others_input_type none --obstacles_input_type dynamic_radii \
--datasets_training "dynObs10quad1_1 dynObs10quad1_2 dynObs10quad1_3" --datasets_validation "dynObs10quad1_4" --datasets_testing "dynObs10quad1_4"

python3 src/master_script_custom.py --model_name onlyEllipsoidObstaclesRNN --model_number 23 \
--others_input_type none --obstacles_input_type dynamic_radii_relvel \
--datasets_training "dynObs10quad1_1 dynObs10quad1_2 dynObs10quad1_3" --datasets_validation "dynObs10quad1_4" --datasets_testing "dynObs10quad1_4"

python3 src/master_script_custom.py --model_name onlyEllipsoidObstaclesRNN --model_number 24 \
--others_input_type none --obstacles_input_type dynamic_points6 \
--datasets_training "dynObs10quad1_1 dynObs10quad1_2 dynObs10quad1_3" --datasets_validation "dynObs10quad1_4" --datasets_testing "dynObs10quad1_4"

python3 src/master_script_custom.py --model_name onlyEllipsoidObstaclesRNN --model_number 25 \
--others_input_type none --obstacles_input_type dynamic_points6_relvel \
--datasets_training "dynObs10quad1_1 dynObs10quad1_2 dynObs10quad1_3" --datasets_validation "dynObs10quad1_4" --datasets_testing "dynObs10quad1_4"



notify-send "Script has finished running"