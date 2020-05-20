import sys, os, argparse, csv
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# sys.path.append("../") # This line is needed if we want to import modules using "src." at the beginning

def parse_args(defaults):
    parser = argparse.ArgumentParser(description="Drone Prediction Model parameters")
    
    #### Model selection ####
    parser.add_argument('--model_name', type=str, default=defaults['model_name'])
    parser.add_argument('--model_number', type=int, default=defaults['model_number'])
    # parser.add_argument(['-m', '--message'], type=str, default="") # TODO: This raises an error, fix it. Maybe it should be a space instead of an empty string.
    
    parser.add_argument('--transfer_learning_others', type=str2bool, default=defaults['TRANSFER_LEARNING_OTHERS'])
    parser.add_argument('--learning_rate_others_encoder', type=float, default=defaults['learning_rate_others_encoder'])
    parser.add_argument('--model_name_others_encoder', type=str, default=defaults['model_name_others_encoder'])
    parser.add_argument('--model_number_others_encoder', type=str, default=defaults['model_number_others_encoder'])

    parser.add_argument('--transfer_learning_obstacles', type=str2bool, default=defaults['TRANSFER_LEARNING_OBSTACLES'])
    parser.add_argument('--learning_rate_obstacles_encoder', type=float, default=defaults['learning_rate_obstacles_encoder'])
    parser.add_argument('--model_name_obstacles_encoder', type=str, default=defaults['model_name_obstacles_encoder'])
    parser.add_argument('--model_number_obstacles_encoder', type=str, default=defaults['model_number_obstacles_encoder'])
    
    parser.add_argument('--raw_data_dir', type=str, default=defaults['raw_data_dir'])
    parser.add_argument('--tfrecord_data_dir', type=str, default=defaults['tfrecord_data_dir'])
    parser.add_argument('--trained_models_dir', type=str, default=defaults['trained_models_dir'])
    
    #### Script options ####
    parser.add_argument('--train', help="Boolean to determine whether to train or to test", type=str2bool, default=defaults['TRAIN'])
    parser.add_argument('--warmstart', type=str2bool, default=defaults['WARMSTART'])
    parser.add_argument('--summary', type=str2bool, default=defaults['SUMMARY'])
    parser.add_argument('--display', type=str2bool, default=defaults['DISPLAY'])
    parser.add_argument('--record', type=str2bool, default=defaults['RECORD'])
    parser.add_argument('--export_plotting_data', type=str2bool, default=defaults['EXPORT_PLOTTING_DATA'])
    parser.add_argument('--n_frames', type=int, default=defaults['N_FRAMES'])
    parser.add_argument('--dt', help="Sampling rate, used to determine the FPS of the animation", type=str2bool, default=defaults['DT'])
    parser.add_argument('--plot_goals', type=str2bool, default=defaults['PLOT_GOALS'])
    parser.add_argument('--plot_ellipsoids', type=str2bool, default=defaults['PLOT_ELLIPSOIDS'])
    
    #### Dataset selection ####
    parser.add_argument('--datasets_training', help="Dataset names to be used for training, separated by spaces", type=str, default=defaults['datasets_training'])
    parser.add_argument('--datasets_validation', help="Dataset names to be used for training, separated by spaces", type=str, default=defaults['datasets_validation'])
    parser.add_argument('--datasets_testing', help="Dataset names to be used for training, separated by spaces", type=str, default=defaults['datasets_test'])
    
    #### Training parameters ####
    parser.add_argument('--max_epochs', type=int, default=defaults['MAX_EPOCHS'])
    parser.add_argument('--max_steps', type=int, default=defaults['MAX_STEPS'])
    parser.add_argument('--train_patience', type=int, default=defaults['TRAIN_PATIENCE'])
    parser.add_argument('--batch_size', type=int, default=defaults['BATCH_SIZE'])

    #### Network architecture ####
    # parser.add_argument('--input_types', help="String with the list of input types", type=str, default=defaults['input_types'])
    parser.add_argument('--query_input_type', help="String with the data type for the query robot input", type=str, default=defaults['query_input_type'])
    parser.add_argument('--others_input_type', help="String with the data type for the other robots input", type=str, default=defaults['others_input_type'])
    parser.add_argument('--obstacles_input_type', help="String with the data type for the obstacles input", type=str, default=defaults['obstacles_input_type'])
    parser.add_argument('--target_type', help="String with the data type for the network target", type=str, default=defaults['target_type'])
    parser.add_argument('--past_horizon', type=int, default=defaults['past_horizon'])
    parser.add_argument('--prediction_horizon', type=int, default=defaults['prediction_horizon'])
    parser.add_argument('--test_prediction_horizons', type=str, default=defaults['test_prediction_horizons'])
    parser.add_argument('--separate_goals', type=str2bool, default=defaults['separate_goals'])
    parser.add_argument('--separate_obstacles', type=str2bool, default=defaults['separate_obstacles'])
    parser.add_argument('--remove_stuck_quadrotors', type=str2bool, default=defaults['remove_stuck_quadrotors'])
    
    # Encoder sizes
    parser.add_argument('--size_query_agent_state', help="Size of the query agent dynamics encoder", type=int, default=defaults['size_query_agent_state'])
    parser.add_argument('--size_other_agents_state', help="Size of the other agents dynamics encoder", type=int, default=defaults['size_other_agents_state'])
    parser.add_argument('--size_other_agents_bilstm', help="Size of the other agents interaction encoder", type=int, default=defaults['size_other_agents_bilstm'])
    parser.add_argument('--size_obstacles_fc_layer', help="Size of the static obstacles encoders", type=int, default=defaults['size_obstacles_fc_layer'])
    parser.add_argument('--size_obstacles_bilstm', help="Size of the obstacles interaction encoder", type=int, default=defaults['size_obstacles_bilstm'])
    parser.add_argument('--size_action_encoding', help="Size of the robot candidate action encoder", type=int, default=defaults['size_action_encoding'])
    
    # Decoder sizes
    parser.add_argument('--size_decoder_lstm', help="Size of the recurrent decoder", type=int, default=defaults['size_decoder_lstm'])
    parser.add_argument('--size_fc_layer', help="Size of the hidden fully connected layer before the output", type=int, default=defaults['size_fc_layer'])

    args = parser.parse_args()

    return args

def parse_input_types(input_types_string):
	type_list = []

	for input_type in input_types_string.split():
		subtype_list = []
		for types in input_type.split("_"):
			subtype_list.append(types)
		type_list.append(subtype_list)
	return type_list

def parse_dataset_names(dataset_names_string):
    dataset_list = []
    for dataset in dataset_names_string.split():
        dataset_list.append(dataset)
    return dataset_list

def model_selector(args):
    if args.model_name == "varyingNQuadsRNN_v2":
        from models.varyingNQuadsRNN_v2 import FullModel
    elif args.model_name == "dynamicEllipsoidObstaclesRNN" or args.model_name == "staticEllipsoidObstaclesRNN":
        from models.dynamicEllipsoidObstaclesRNN import FullModel
    elif args.model_name == "onlyEllipsoidObstaclesRNN":
        from models.onlyEllipsoidObstaclesRNN import FullModel
    elif args.model_name == "dynamicEllipsoidObstaclesRNN_commonInput":
        from models.dynamicEllipsoidObstaclesRNN_commonInput import FullModel
    elif args.model_name == "dynamicEllipsoidObstaclesRNN_subInputs":
        from models.dynamicEllipsoidObstaclesRNN_subInputs import FullModel
    else:
        raise Exception("Unrecognised model name")

    model = FullModel(args)
    model.compile(loss=model.loss_object, optimizer=model.optimizer)

    return model

def combine_args(parsed_args, stored_args):
    # Combine parsed and stored args in order to ensure the correct network architecture
    new_args = deepcopy(parsed_args)
    
    new_args.query_input_type = stored_args.query_input_type
    new_args.others_input_type = stored_args.others_input_type
    new_args.obstacles_input_type = stored_args.obstacles_input_type
    new_args.target_type = stored_args.target_type
    
    new_args.past_horizon = stored_args.past_horizon
    # new_args.prediction_horizon = stored_args.prediction_horizon

    new_args.separate_goals = stored_args.separate_goals
    new_args.separate_obstacles = stored_args.separate_obstacles
    new_args.remove_stuck_quadrotors = stored_args.remove_stuck_quadrotors

    # Encoder sizes
    new_args.size_query_agent_state = stored_args.size_query_agent_state
    new_args.size_other_agents_state = stored_args.size_other_agents_state
    new_args.size_other_agents_bilstm = stored_args.size_other_agents_bilstm
    new_args.size_obstacles_fc_layer = stored_args.size_obstacles_fc_layer
    new_args.size_obstacles_bilstm = stored_args.size_obstacles_bilstm
    new_args.size_action_encoding = stored_args.size_action_encoding

    # Decoder sizes
    new_args.size_decoder_lstm = stored_args.size_decoder_lstm
    new_args.size_decoder_lstm = stored_args.size_decoder_lstm
    
    return new_args

def save_model_summary(args, train_loss, val_loss, test_loss, termination_type, train_time):
    """
    Store model summary
    """
    summary_file = "trained_models_summary.csv"
    
    new_row = [
        args.model_name,
        args.model_number,
        train_loss,
        val_loss,
        test_loss,
        args.past_horizon,
        args.prediction_horizon,
        args.query_input_type,
        args.others_input_type,
        args.obstacles_input_type,
        args.target_type,
        args.max_epochs,
        args.max_steps,
        args.train_patience,
        args.batch_size,
        termination_type,
        args.datasets_training,
        args.datasets_validation,
        args.datasets_testing,
        args.separate_goals,
        args.size_query_agent_state,
        args.size_other_agents_state,
        args.size_other_agents_bilstm,
        args.size_obstacles_fc_layer,
        args.size_obstacles_bilstm,
        args.size_decoder_lstm,
        args.size_fc_layer,
        train_time,
        datetime.now().strftime("%d/%m/%Y %H:%M") 
    ]
    
    # If the file does not exist already, create first row
    if not os.path.isfile(summary_file):
        with open(summary_file, 'w') as new_file:
            first_row = [
                "Model name", 
                "Model number",
                "Training loss",
                "Validation loss",
                "Test loss",
                "Past horizon",
                "Prediction horizon",
                "Query input type",
                "Others input type",
                "Obstacle input type",
                "Target type",
                "Max epochs",
                "Max steps",
                "Train patience",
                "Batch size",
                "Termination",
                "Training datasets",
                "Validation datasets",
                "Test datasets",
                "Separate goals",
                "Size query agent state",
                "Size other agents state",
                "Size other agents biLSTM",
                "Size obstacles FC layer",
                "Size obstacles biLSTM",
                "Size decoder LSTM",
                "Size fc layer",
                "Training time (sec)",
                "Date"
            ]
            
            writer = csv.writer(new_file)
            writer.writerow(first_row)

    
    with open(summary_file, 'r') as in_file:
        reader = csv.reader(in_file)
        lines = list(reader)
        if len(lines) == 1:
            lines.append(new_row)
        else:
            rewrite = False
            new_model = True
            for idx in range(len(lines)):
                if lines[idx][0] == new_row[0]: # If there are already models with the same name
                    if new_model == True:
                        new_idx = idx
                        new_model = False

                    if lines[idx][1] == new_row[1]: # If experiment number is the same, substitute data
                        lines[idx] = new_row
                        rewrite = True
                    elif int(lines[idx][1]) < new_row[1]: # Else store the idx of the last data entry with an experiment number lower than the one of the new data
                        new_idx = idx+1

            if new_model:
                lines.append(new_row)
            elif not rewrite:
                lines.insert(new_idx, new_row)
    
    with open(summary_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)
    
def save_fde_summary(args, fde_list):
    summary_file = "fde_summary.csv"

    new_row = [
        args.model_name,
        args.model_number,
        args.past_horizon,
        args.prediction_horizon,
        args.test_prediction_horizons,
        fde_list[0]["position"],
        fde_list[1]["position"],
        fde_list[2]["position"],
        fde_list[3]["position"],
        fde_list[0]["velocity"],
        fde_list[1]["velocity"],
        fde_list[2]["velocity"],
        fde_list[3]["velocity"],
        args.query_input_type,
        args.others_input_type,
        args.obstacles_input_type,
        args.target_type,
        args.datasets_testing,
        args.size_query_agent_state,
        args.size_other_agents_state,
        args.size_other_agents_bilstm,
        args.size_obstacles_fc_layer,
        args.size_obstacles_bilstm,
        args.size_decoder_lstm,
        args.size_fc_layer,
        datetime.now().strftime("%d/%m/%Y %H:%M") 
    ]
    
    prediction_horizons = []
    for prediction_horizon in args.test_prediction_horizons.split(" "):
        prediction_horizons.append(int(prediction_horizon))
    
    # If the file does not exist already, create first row
    if not os.path.isfile(summary_file):
        with open(summary_file, 'w') as new_file:
            first_row = [
                "Model name", 
                "Model number",
                "Training past horizon",
                "Training prediction horizon",
                "Test prediction horizons",
                f"FDE@%d steps (pos)" % prediction_horizons[0],
                f"FDE@%d steps (pos)" % prediction_horizons[1],
                f"FDE@%d steps (pos)" % prediction_horizons[2],
                f"FDE@%d steps (pos)" % prediction_horizons[3],
                f"FDE@%d steps (vel)" % prediction_horizons[0],
                f"FDE@%d steps (vel)" % prediction_horizons[1],
                f"FDE@%d steps (vel)" % prediction_horizons[2],
                f"FDE@%d steps (vel)" % prediction_horizons[3],
                "Query input type",
                "Others input type",
                "Obstacle input type",
                "Target type",
                "Test datasets",
                "Size query agent state",
                "Size other agents state",
                "Size other agents biLSTM",
                "Size obstacles FC layer",
                "Size obstacles biLSTM",
                "Size decoder LSTM",
                "Size fc layer",
                "Date"
            ]

            writer = csv.writer(new_file)
            writer.writerow(first_row)
            
    with open(summary_file, 'r') as in_file:
        reader = csv.reader(in_file)
        lines = list(reader)
        if len(lines) == 1:
            lines.append(new_row)
        else:
            rewrite = False
            new_model = True
            for idx in range(len(lines)):
                if lines[idx][0] == new_row[0]: # If there are already models with the same name
                    if new_model == True:
                        new_idx = idx
                        new_model = False

                    if lines[idx][1] == new_row[1]: # If experiment number is the same, substitute data
                        lines[idx] = new_row
                        rewrite = True
                    elif int(lines[idx][1]) < new_row[1]: # Else store the idx of the last data entry with an experiment number lower than the one of the new data
                        new_idx = idx+1

            if new_model:
                lines.append(new_row)
            elif not rewrite:
                lines.insert(new_idx, new_row)

    with open(summary_file, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)

def save_full_model_summary(args, last_train_loss, train_loss, val_loss, test_loss, termination_type, train_time, fde_list):
    """
    Store model summary
    """
    summary_file = "trained_models_summary_v2.csv"
    
    new_row = [
        args.model_name,
        args.model_number,
        args.past_horizon,
        args.prediction_horizon,
        args.test_prediction_horizons,
        last_train_loss,
        train_loss,
        val_loss,
        test_loss,
        fde_list[0]["position"],
        fde_list[1]["position"],
        fde_list[2]["position"],
        fde_list[3]["position"],
        fde_list[0]["velocity"],
        fde_list[1]["velocity"],
        fde_list[2]["velocity"],
        fde_list[3]["velocity"],
        args.query_input_type,
        args.others_input_type,
        args.obstacles_input_type,
        args.target_type,
        args.max_epochs,
        args.max_steps,
        args.train_patience,
        args.batch_size,
        termination_type,
        args.datasets_training,
        args.datasets_validation,
        args.datasets_testing,
        args.separate_goals,
        args.separate_obstacles,
        args.size_query_agent_state,
        args.size_other_agents_state,
        args.size_other_agents_bilstm,
        args.size_obstacles_fc_layer,
        args.size_obstacles_bilstm,
        args.size_decoder_lstm,
        args.size_fc_layer,
        train_time,
        datetime.now().strftime("%d/%m/%Y %H:%M") 
    ]
    
    prediction_horizons = []
    for prediction_horizon in args.test_prediction_horizons.split(" "):
        prediction_horizons.append(int(prediction_horizon))
    
    # If the file does not exist already, create first row
    if not os.path.isfile(summary_file):
        with open(summary_file, 'w') as new_file:
            first_row = [
                "Model name", 
                "Model number",
                "Past horizon",
                "Prediction horizon",
                "Test prediction horizons",
                "Lowest training loss",
                "Saved model training loss",
                "Validation loss",
                "Test loss",
                f"FDE@%d steps (pos)" % prediction_horizons[0],
                f"FDE@%d steps (pos)" % prediction_horizons[1],
                f"FDE@%d steps (pos)" % prediction_horizons[2],
                f"FDE@%d steps (pos)" % prediction_horizons[3],
                f"FDE@%d steps (vel)" % prediction_horizons[0],
                f"FDE@%d steps (vel)" % prediction_horizons[1],
                f"FDE@%d steps (vel)" % prediction_horizons[2],
                f"FDE@%d steps (vel)" % prediction_horizons[3],
                "Query input type",
                "Others input type",
                "Obstacle input type",
                "Target type",
                "Max epochs",
                "Max steps",
                "Train patience",
                "Batch size",
                "Termination",
                "Training datasets",
                "Validation datasets",
                "Test datasets",
                "Separate goals",
                "Separate obstacles",
                "Size query agent state",
                "Size other agents state",
                "Size other agents biLSTM",
                "Size obstacles FC layer",
                "Size obstacles biLSTM",
                "Size decoder LSTM",
                "Size fc layer",
                "Training time (sec)",
                "Date"
            ]
            
            writer = csv.writer(new_file)
            writer.writerow(first_row)

    
    with open(summary_file, 'r') as in_file:
        reader = csv.reader(in_file)
        lines = list(reader)
        # lines = list(filter(None, list(reader))) # To remove empty lines which appear in Windows
        if len(lines) == 1:
            lines.append(new_row)
        else:
            rewrite = False
            new_model = True
            for idx in range(len(lines)):
                if lines[idx][0] == new_row[0]: # If there are already models with the same name
                    if new_model == True:
                        new_idx = idx
                        new_model = False

                    if lines[idx][1] == new_row[1]: # If experiment number is the same, substitute data
                        lines[idx] = new_row
                        rewrite = True
                    elif int(lines[idx][1]) < new_row[1]: # Else store the idx of the last data entry with an experiment number lower than the one of the new data
                        new_idx = idx+1

            if new_model:
                lines.append(new_row)
            elif not rewrite:
                lines.insert(new_idx, new_row)
    
    with open(summary_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(lines)


def get_paths(trained_models_dir, args):
    model_dir = os.path.join(trained_models_dir, args.model_name, str(args.model_number), "")
    parameters_path = os.path.join(model_dir, "model_parameters.pkl")
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.h5")
    recording_dir = os.path.join(model_dir, "Recordings", "")
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    return parameters_path, checkpoint_path, recording_dir




def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


