import os, sys, time
from pathlib import Path
from utils.data_handler_v3_tfrecord import DataHandler
from utils.plot_utils_v3 import plot_predictions
from utils.model_utils import parse_args, model_selector, combine_args, save_model_summary

import pickle as pkl
# from progressbar import progressbar
from tqdm import trange

#### Directory definitions ####
root_dir = os.path.dirname(sys.path[0])
data_master_dir = os.path.join(root_dir, "data", "")
raw_data_dir = os.path.join(data_master_dir, "Raw", "")
tfrecord_data_dir = os.path.join(data_master_dir, "TFRecord", "")
trained_models_dir = os.path.join(root_dir, "trained_models", "")


#### Model selection ####
model_name = "varyingNQuadsRNN_v2"
model_number = 1


#### Script options ####
TRAIN = True
WARMSTART = False

SUMMARY = False # To include a summary of the results of the model in a csv file

# To display and/or record an animation of the test dataset with the trajcetory predictions from the model
DISPLAY = False
RECORD = True
N_FRAMES = 2500 # Number of frames to display/record
DT = 0.05 # Used to compute the FPS of the video
PLOT_GOALS = False # To plot the goals of the quadrotors


#### Datasets selection ####
# datasets_training = "dynamic16quads1\
#                     dynamic16quadsPosExchange"
# datasets_validation = "dynamic16quads2"
# datasets_test = "goalSequence16quads1"

datasets_training = "obsfree_q16_g250_20200428-221906"
datasets_validation = "obsfree_q16_g50_20200428-175033"
datasets_test = "obsfree_q16_g50_20200429-144614"


#### Training parameters ####
MAX_EPOCHS = 15
MAX_STEPS = 1E6
TRAIN_PATIENCE = 5 # Number of epochs before early stopping
BATCH_SIZE = 64 


#### Network architecture ####
# input_types = "vel relpos_vel"
query_input_type = "vel"
others_input_type = "relpos_vel"
obstacle_input_type = "none"
target_type = "vel"

past_horizon = 10
prediction_horizon = 15
separate_goals = True

# Encoder sizes
size_query_agent_state = 256
size_other_agents_state = 256
size_other_agents_bilstm = 256
size_action_encoding = 0

# Decoder sizes
size_decoder_lstm = 512
size_fc_layer = 256


#### Parse args ####
args = parse_args(locals())
model_dir = os.path.join(trained_models_dir, model_name, str(model_number), "")
parameters_path = os.path.join(model_dir, "model_parameters.pkl")
checkpoint_path = os.path.join(model_dir, "model_checkpoint.h5")
Path(model_dir).mkdir(parents=True, exist_ok=True)

# Load model parameters if warmstarting the model or if we are testing it
if args.warmstart or not args.train:
    assert os.path.isfile(checkpoint_path)    
    stored_args = pkl.load( open( parameters_path, "rb" ) )
    args = combine_args(args, stored_args) # To ensure the correct model architecture
else:
    pkl.dump( args, open( parameters_path, "wb" ) )


#### Define data object, construct model, and load model weights if necessary ####
data = DataHandler(args)
# tfdataset_training = data.getTrainingDataset()
# tfdataset_validation = data.getValidationDataset()

model = model_selector(args)
train_loss = float('inf')
val_loss = float('inf')
test_loss = float('inf')
termination_type = "None"

# Load weights if warmstarting the model or if we are testing it
if args.warmstart or not args.train: # TODO: check that this works
    sample_input_batch = data.getSampleInputBatch()
    model.call(sample_input_batch)
    model.load_weights(checkpoint_path)


#### Training loop ####
if args.train:    
    # TODO: Create a loop with trange for the steps in an epoch. For this, we need to know the number of samples in the training dataset, which we don't know right now
    # steps_per_epoch = 
    step = 1
    
    best_loss = float("inf")
    patience_counter = 0
    
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
        train_loss = float(model.train_loss.result())
        
        # Validation step
        if data.tfdataset_validation != None:
            for batch in data.tfdataset_validation:
                model.val_step(batch)
            val_loss = float(model.val_loss.result())
            curr_loss = val_loss
            loss_type = "Validation"
        else:
            val_loss = float('inf')
            curr_loss = train_loss
            loss_type = "Training"
        
        ellapsed_time = time.time() - start_time
        print("\n\nEpoch: %d/%d, Steps: %d/%d, Train loss: %.4e, Validation loss: %.4e, Epoch time: %.1f sec" % (epoch+1, args.max_epochs, step, args.max_steps, train_loss, val_loss, ellapsed_time))
        
        if curr_loss < best_loss:
            print(f"%s loss improved, saving new best model" % loss_type)
            best_loss = curr_loss
            best_model = model
            patience_counter = 0
            best_model.save_weights(checkpoint_path)
        else:
            print(f"%s loss did not improve" % loss_type)
            patience_counter += 1        
        
        if step > args.max_steps:
            # Message already printed before
            break
        
        if patience_counter >= args.train_patience:
            print("Maximum patience reached, terminating early")
            termination_type = "Max patience"
            break
        
    model = best_model
    if termination_type == "none":
        termination_type = "Max epochs"

# Set model back to stateless for prediction
if model.stateful:
    for i in range(len(model.layers)):
        model.layers[i].stateful = False

#### Model evaluation ####
# TODO: Test model on testing datasets
if args.datasets_testing != []:
    model.val_loss.reset_states()
    model.val_step(batch)
    pass

# TODO: Implement summary writing function
# Get summary of the model performance and store it in a CSV file
if args.summary:
    save_model_summary(args, train_loss, val_loss, test_loss, termination_type)

# Plot and/or record animation to evaluate the network's prediction performance 
if args.display or args.record:
    for dataset_idx in range(len(data.datasets_testing)):
        plotting_data = data.getPlottingData(model, dataset_idx)
        
        # Path the recording will be stored if this option has been selected
        dataset_name = data.datasets_testing[dataset_idx]
        recording_path = os.path.join(model_dir, "Recordings", dataset_name + ".mp4")

        # Create animation
        plot_predictions(plotting_data, args, recording_path)

print("Master script finished")