import os, sys, time
import tensorflow as tf
from copy import deepcopy
from pathlib import Path
from shutil import rmtree 
from datetime import datetime
from utils.data_handler_v3_tfrecord import DataHandler
from utils.plot_utils_v3 import plot_predictions
from utils.model_utils import parse_args, model_selector, combine_args, save_model_summary, save_fde_summary, save_full_model_summary, get_paths

import pickle as pkl
from tqdm import trange

# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#### Directory definitions ####
root_dir = os.path.dirname(sys.path[0])
data_master_dir = os.path.join(root_dir, "data", "")
raw_data_dir = os.path.join(data_master_dir, "Raw", "")
tfrecord_data_dir = os.path.join(data_master_dir, "TFRecord", "")
trained_models_dir = os.path.join(root_dir, "trained_models", "")


#### Model selection ####
model_name = "varyingNQuadsRNN_v2"
# model_name = "staticEllipsoidObstaclesRNN" # Same as dynamic, only the name changes
# model_name = "dynamicEllipsoidObstaclesRNN"
# model_name = "dynamicEllipsoidObstaclesRNN_regularization"
# model_name = "dynamicEllipsoidObstaclesRNN_layerNorm"
# model_name = "dynamicEllipsoidObstaclesRNN_subInputs"
# model_name = "dynamicEllipsoidObstaclesRNN_commonInput"
# model_name = "onlyEllipsoidObstaclesRNN"
# model_name = "onlyEllipsoidObstaclesRNN_RBF"
model_number = 0

TRANSFER_LEARNING_OTHERS = False
learning_rate_others_encoder = 0 # TODO: Set up an option to optimize the weigths of the others encoder even when using transfer learning
model_name_others_encoder = "varyingNQuadsRNN_v2"
model_number_others_encoder = 17

TRANSFER_LEARNING_OBSTACLES = False
learning_rate_obstacles_encoder = 0 # TODO: Set up an option to optimize the weigths of the obstacles encoder even when using transfer learning
model_name_obstacles_encoder = "onlyEllipsoidObstaclesRNN"
model_number_obstacles_encoder = 20

#### Script options ####
TRAIN = True
WARMSTART = False

SUMMARY = False # To include a summary of the results of the model in a csv file

# To display and/or record an animation of the test dataset with the trajcetory predictions from the model
DISPLAY = False
RECORD = False
EXPORT_PLOTTING_DATA = False
N_FRAMES = 2000 # Number of frames to display/record
DT = 0.05 # Used to compute the FPS of the video
PLOT_GOALS = True # To plot the goals of the quadrotors
PLOT_ELLIPSOIDS = False

#### Datasets selection ####
# datasets_training = "goalSequence1\
#                     goalSequence2\
#                     goalSequence3\
#                     goalSequence4"
# datasets_validation = "goalSequence5"
# datasets_test = datasets_validation

# datasets_training = "dynamic16quads1\
#                     dynamic16quadsPosExchange"
# datasets_validation = "dynamic16quads2"
# datasets_test = datasets_validation

# datasets_training = "staticObs6quad10_1"
# datasets_validation = "staticObs6quad10_2"
# datasets_test = datasets_validation

datasets_training = "dynObs10quad10_1 dynObs10quad10_5 dynObs10quad10_6 dynObs10quad10_bugged dynObs10quad10_2"
datasets_validation = "dynObs10quad10_3 dynObs10quad10_4"
datasets_test = datasets_validation

# datasets_training = "dynObs10quad10_2"
# datasets_validation = "dynObs10quad10_3"
# datasets_test = datasets_validation

# datasets_training = "dynObs10quad10_small"
# datasets_validation = "dynObs10quad10_small"
# datasets_test = datasets_validation

# datasets_training = "dynObs10quad1_1\
#                     dynObs10quad1_2\
#                     dynObs10quad1_3"
# datasets_validation = "dynObs10quad1_4"
# datasets_test = datasets_validation


#### Training parameters ####
MAX_EPOCHS = 15
MAX_STEPS = 1E6
TRAIN_PATIENCE = 4 # Number of epochs before early stopping
BATCH_SIZE = 256 


#### Network architecture ####
# Network types are unused so far
query_input_type = "vel" # {vel}
others_input_type = "relpos_vel" # {none, relpos_vel, relpos_relvel}
obstacles_input_type = "dynamic" # {none, static, dynamic, dynamic_radii, dynamic_points6} (dynamic options can also use _relvel)
target_type = "vel" # {vel}

past_horizon = 10
prediction_horizon = 10
# test_prediction_horizons = "none"
test_prediction_horizons = "5 10 15 20"
separate_goals = True # To make sure that training trajectories keep goal position constant
separate_obstacles = False # Only makes sense if using multiple steps of the obstacle state
if obstacles_input_type != "none" and others_input_type != "none": # Quadrotors get stuck when there are both obstacles and other quadrotors
    remove_stuck_quadrotors = True
else:
    remove_stuck_quadrotors = False

# Encoder sizes
size_query_agent_state = 64 # 256
size_other_agents_state = 64 # 256
size_other_agents_bilstm = 64 # 256
size_obstacles_fc_layer = 32 # 64
size_obstacles_bilstm = 32 # 64
size_action_encoding = 0 # 0

# Decoder sizes
size_decoder_lstm = 128 # 512
size_fc_layer = 64 # 256


#### Parse args ####
args = parse_args(locals())
if args.model_number == -1:
    model_parent_dir = os.path.join(trained_models_dir, args.model_name)
    if os.path.isdir(model_parent_dir):
        trained_model_numbers = os.listdir(model_parent_dir)
        if args.train: # Use next available model number
            args.model_number = int(trained_model_numbers[-1]) + 1
        else: # Use model with the highest model number
            args.model_number = int(trained_model_numbers[-1])
    else:
        args.model_number = 0

print(f"\n{bcolors.BOLD}{bcolors.HEADER}[%s] Starting master script. Model name: %s, model number: %d{bcolors.ENDC}" % (datetime.now().strftime("%d-%m-%y %H:%M:%S"), args.model_name, args.model_number))

parameters_path, checkpoint_path, recording_dir = get_paths(trained_models_dir, args)

# Load model parameters if warmstarting the model or if we are testing it
if args.warmstart or not args.train:
    assert os.path.isfile(checkpoint_path)    
    stored_args = pkl.load( open( parameters_path, "rb" ) )
    args = combine_args(args, stored_args) # To ensure the correct model architecture
else:
    if os.path.isfile(checkpoint_path):
        print(f"{bcolors.FAIL}WARNING: Rewriting previously trained model!{bcolors.ENDC}")
        Path(checkpoint_path).unlink() # Delete checkpoint if we are training a new model with an already existing experiment number to avoid problems if the training loop were to be prematurely stopped
    
    if os.path.isdir(recording_dir): # Delete recordings if we are training a new model with an already existing experiment number
        rmtree(recording_dir) 
    
    pkl.dump( args, open( parameters_path, "wb" ) )


#### Define data object, construct model, and load model weights if necessary ####
data = DataHandler(args)

model = model_selector(args)
train_time = 0
train_loss = float('inf')
val_loss = float('inf')
test_loss = float('inf')
termination_type = "None"

# Load weights if warmstarting the model or if we are testing it
if args.warmstart or args.transfer_learning_others or args.transfer_learning_obstacles: # TODO: check that this works
    sample_input_batch = data.getSampleInputBatch()
    model.call(sample_input_batch)
    
    if args.warmstart:
        print(f"\n{bcolors.OKGREEN}[%s] Loading model %s, model number %d{bcolors.ENDC}\n" % (datetime.now().strftime("%d-%m-%y %H:%M:%S"), args.model_name, args.model_number))
        
        model.load_weights(checkpoint_path)


    if args.transfer_learning_others:
        print(f"\n{bcolors.OKGREEN}[%s] Other agents encoder transfer learning from model %s, model number %d{bcolors.ENDC}\n" % (datetime.now().strftime("%d-%m-%y %H:%M:%S"), args.model_name_others_encoder, args.model_number_others_encoder))
        
        args_others = deepcopy(args)
        args_others.model_name = args.model_name_others_encoder
        args_others.model_number = args.model_number_others_encoder
        parameters_path_others, checkpoint_path_others, _ = get_paths(trained_models_dir, args_others)

        stored_args = pkl.load( open( parameters_path_others, "rb" ) )
        assert args_others.past_horizon == stored_args.past_horizon and \
            args_others.size_obstacles_fc_layer == stored_args.size_obstacles_fc_layer and \
            args_others.size_obstacles_bilstm == stored_args.size_obstacles_bilstm and\
            args_others.others_input_type == stored_args.others_input_type,\
            "Arguments for transfer learning of the other agents encoder are not valid for the selected trained model"

        model_others = model_selector(args_others)
        model_others.call(sample_input_batch)
        model_others.load_weights(checkpoint_path_others)

        for idx in range(len(model_others.other_quads_encoder._layers)):
            extracted_weights = model_others.other_quads_encoder._layers[idx].get_weights()
            model.other_quads_encoder._layers[idx].set_weights(extracted_weights)

        model.other_quads_encoder.trainable = False


    if args.transfer_learning_obstacles:
        print(f"\n{bcolors.OKGREEN}[%s] Obstacles encoder transfer learning from model %s, model number %d{bcolors.ENDC}\n" % (datetime.now().strftime("%d-%m-%y %H:%M:%S"), args.model_name_obstacles_encoder, args.model_number_obstacles_encoder))

        args_obstacles = deepcopy(args)
        args_obstacles.model_name = args.model_name_obstacles_encoder
        args_obstacles.model_number = args.model_number_obstacles_encoder
        parameters_path_obstacles, checkpoint_path_obstacles, _ = get_paths(trained_models_dir, args_obstacles)

        stored_args = pkl.load( open( parameters_path_obstacles, "rb" ) )
        # assert args_obstacles == combine_args(args_obstacles, stored_args), \
        assert args_obstacles.past_horizon == stored_args.past_horizon and \
            args_obstacles.size_obstacles_fc_layer == stored_args.size_obstacles_fc_layer and \
            args_obstacles.size_obstacles_bilstm == stored_args.size_obstacles_bilstm and\
            args_obstacles.obstacles_input_type == stored_args.obstacles_input_type, \
            "Arguments for transfer learning of the obstacles encoder are not valid for the selected trained model"

        model_obstacles = model_selector(args_obstacles)
        model_obstacles.call(sample_input_batch)
        model_obstacles.load_weights(checkpoint_path_obstacles)
            
        for idx in range(len(model_obstacles.obs_encoder._layers)):
            extracted_weights = model_obstacles.obs_encoder._layers[idx].get_weights()
            model.obs_encoder._layers[idx].set_weights(extracted_weights)

        model.obs_encoder.trainable = False
    

#### Training loop ####
if args.train:    
    step = 1
    
    best_loss = float("inf")
    patience_counter = 0
    time_training_init = time.time()
    print(f"\n{bcolors.OKGREEN}[%s] Starting training of model %s, model number %d{bcolors.ENDC}\n" % (datetime.now().strftime("%d-%m-%y %H:%M:%S"), args.model_name, args.model_number))
    
    for epoch in trange(args.max_epochs):
        # new_epoch = False
        # validation_finished = False
        model.train_loss.reset_states()
        model.val_loss.reset_states()
        start_time = time.time()
        
        # Training epoch
        for batch in data.tfdataset_training:
            model.train_step(batch)
            step += 1
            if step > args.max_steps:
                print("Max number of steps reached, terminating early")
                termination_type = "Max steps"
                break
        train_loss = model.train_loss.result().numpy()
        
        # Validation step
        if data.tfdataset_validation is not None:
            for batch in data.tfdataset_validation:
                model.val_step(batch)
            val_loss = model.val_loss.result().numpy()
            curr_loss = val_loss
            loss_type = "Validation"
        else:
            val_loss = float('inf')
            curr_loss = train_loss
            loss_type = "Training"
        
        ellapsed_time = time.time() - start_time
        print(f"\n\n{bcolors.OKBLUE}Epoch: %d/%d, Steps: %d/%d, Train loss: %.4e, Validation loss: %.4e, Epoch time: %.1f sec{bcolors.ENDC}" % (epoch+1, args.max_epochs, step, args.max_steps, train_loss, val_loss, ellapsed_time))
        
        if curr_loss < best_loss:
            print(f"{bcolors.OKGREEN}%s loss improved, saving new best model{bcolors.ENDC}" % loss_type)
            model.save_weights(checkpoint_path)
            best_loss = curr_loss
            patience_counter = 0
        else:
            print(f"{bcolors.FAIL}%s loss did not improve{bcolors.ENDC}" % loss_type)
            patience_counter += 1        
        
        if step > args.max_steps:
            # Message already printed before
            break
        
        if patience_counter >= args.train_patience:
            print("Maximum patience reached, terminating early")
            termination_type = "Max patience"
            break
        
    if termination_type.lower() == "none":
        termination_type = "Max epochs"
        
    train_time = time.time() - time_training_init

# Retrieve best model
model = model_selector(args)
if 'sample_input_batch' not in locals():
    sample_input_batch = data.getSampleInputBatch()
model.call(sample_input_batch)
model.load_weights(checkpoint_path)

# Set model back to stateless for prediction
if model.stateful:
    for i in range(len(model.layers)):
        model.layers[i].stateful = False

#### Model evaluation ####
if args.summary:
    print(f"\n{bcolors.OKGREEN}[%s] Saving summary of the model{bcolors.ENDC}\n" % datetime.now().strftime("%d-%m-%y %H:%M:%S"))

    # Retrieve best validation loss and corresponding training loss
    last_train_loss = train_loss
    train_loss = model.train_loss.result().numpy()
    val_loss = model.val_loss.result().numpy()

    if args.datasets_testing != []:
        model.val_loss.reset_states()
        for batch in data.tfdataset_testing:
            model.val_step(batch)
        test_loss = float(model.val_loss.result())
        print(f"Test loss: %e\n" % test_loss)
        
        fde_list = []
        if len(args.test_prediction_horizons.split(" ")) > 1:
            print(f"{bcolors.OKBLUE}[%s] Evaluating FDE for different prediction horizons{bcolors.ENDC}\n" % datetime.now().strftime("%d-%m-%y %H:%M:%S"))
            test_prediction_horizon_list = []
            for pred_horizon in test_prediction_horizons.split(" "):
                test_prediction_horizon_list.append(int(pred_horizon))

            test_args = deepcopy(args)
            test_args.prediction_horizon = test_prediction_horizon_list[-1]

            trained_model = model_selector(test_args)
            sample_test_input_batch = data.getSampleInputBatch(dataset_type="test")
            trained_model.call(sample_test_input_batch)
            trained_model.load_weights(checkpoint_path)

            for batch in data.tfdataset_fde_testing:
                trained_model.testFDE_step(batch)

            for position_FDE, velocity_FDE in zip(trained_model.position_L2_errors, trained_model.velocity_L2_errors):
                fde_list.append({"position": position_FDE.result().numpy(), "velocity": velocity_FDE.result().numpy()})
        else:
            fde = {"position":float('inf'), "velocity":float('inf')}
            for _ in range(4):
                fde_list.append(fde)

    save_full_model_summary(args, last_train_loss, train_loss, val_loss, test_loss, termination_type, train_time, fde_list)

# Plot and/or record animation to evaluate the network's prediction performance 
if args.display or args.record or args.export_plotting_data:
    print(f"\n{bcolors.OKGREEN}[%s] Getting test animation{bcolors.ENDC}\n" % datetime.now().strftime("%d-%m-%y %H:%M:%S"))

    for dataset_idx in range(len(data.datasets_testing)):
        plotting_data = data.getPlottingData(model, dataset_idx, quads_to_plot = -1) # quads_to_plot = -1 to plot predictions for all quadrotors
        
        # Path the recording will be stored if this option has been selected
        dataset_name = data.datasets_testing[dataset_idx]
        recording_path = os.path.join(recording_dir, dataset_name + ".mp4")
        export_plotting_data_path = os.path.join(recording_dir, dataset_name + ".mat")

        if args.export_plotting_data:
            data.savePlottingData(export_plotting_data_path, plotting_data, max_samples = args.n_frames)

        if args.display or args.record:
            # Create animation
            plot_predictions(plotting_data, args, recording_path)

print(f"\n{bcolors.BOLD}{bcolors.OKGREEN}[%s] Master script finished{bcolors.ENDC}" % datetime.now().strftime("%d-%m-%y %H:%M:%S"))