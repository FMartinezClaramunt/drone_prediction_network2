import sys
import argparse

sys.path.append("../")

def parse_args(defaults):
    parser = argparse.ArgumentParser(description="Drone Prediction Model parameters")
    
    #### Model selection ####
    parser.add_argument('--model_name', type=str, default=defaults['model_name'])
    parser.add_argument('--model_number', type=int, default=defaults['model_number'])
    
    #### Script options ####
    parser.add_argument('--train', help="Boolean to determine whether to train or to test", type=str2bool, default=defaults['TRAIN'])
    parser.add_argument('--warmstart', type=str2bool, default=defaults['WARMSTART'])
    parser.add_argument('--summary', type=str2bool, default=defaults['SUMMARY'])
    parser.add_argument('--record', type=str2bool, default=defaults['RECORD'])
    parser.add_argument('--display', type=str2bool, default=defaults['DISPLAY'])
    parser.add_argument('--n_frames', type=int, default=defaults['N_FRAMES'])
    parser.add_argument('--dt', help="Sampling rate, used to determine the FPS of the animation", type=str2bool, default=defaults['DT'])
    parser.add_argument('--plot_goals', type=str2bool, default=defaults['PLOT_GOALS'])
    
    #### Dataset selection ####
    parser.add_argument('--datasets_training', help="Dataset names to be used for training, separated by spaces", type=str, default=defaults['datasets_training'])
    parser.add_argument('--datasets_validation', help="Dataset names to be used for training, separated by spaces", type=str, default=defaults['datasets_validation'])
    parser.add_argument('--datasets_testing', help="Dataset names to be used for training, separated by spaces", type=str, default=defaults['datasets_validation'])
    
    #### Training parameters ####
    parser.add_argument('--max_epochs', type=int, default=defaults['MAX_EPOCHS'])
    parser.add_argument('--max_steps', type=int, default=defaults['MAX_STEPS'])
    parser.add_argument('--train_patience', type=int, default=defaults['TRAIN_PATIENCE'])
    parser.add_argument('--batch_size', type=int, default=defaults['BATCH_SIZE'])

    #### Network architecture ####
    parser.add_argument('--input_types', help="String with the list of input types", type=str, default=defaults['input_types'])
    parser.add_argument('--past_horizon', type=str, default=defaults['past_horizon'])
    parser.add_argument('--prediction_horizon', type=str, default=defaults['prediction_horizon'])
    
    # Encoder sizes
    parser.add_argument('--size_query_agent_state', help="Size of the query agent dynamics encoder", type=int, default=defaults['size_query_agent_state'])
    parser.add_argument('--size_other_agents_state', help="Size of the other agentsdynamics encoder", type=int, default=defaults['size_other_agents_state'])
    parser.add_argument('--size_other_agents_bilstm', help="Size of the other agents interaction encoder", type=int, default=defaults['size_other_agents_bilstm'])
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

def parse_mat_dataset_names(dataset_names_string):
    dataset_list = []
    for dataset in dataset_names_string.split():
        dataset_list.append(dataset + ".mat")
    return dataset_list

def model_selector(args):
    if args.model_name == "varyingNQuadsRNN_v2":
        from src.models.varyingNQuadsRNN_v2 import FullModel
    # elif args.model_name == "interactionLSTM":
    #     from src.models.interactionLSTM import FullModel
    else:
        raise Exception("Unrecognised model name")

    model = FullModel(args)
    model.compile(loss=model.loss_object, optimizer=model.optimizer)

    return model

def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


