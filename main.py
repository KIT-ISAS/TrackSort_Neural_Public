import code # code.interact(local=dict(globals(), **locals()))

from DataAssociation import DataAssociation

global_config = {
	'is_loaded' : True,
	'model_path' : 'models/rnn_model_fake_data.h5',
	'distance_threshhold' : 5.0,
	'num_timesteps' : 35, # TODO
	#
	'initial_is_alive_probability' : 0.5,
	'is_alive_decrease' : 0.25,
	'is_alive_increase' : 0.5,
	'batch_size' : 128, # TODO
}

data_association = DataAssociation(global_config)

tracks = data_association.associate_data()

code.interact(local=dict(globals(), **locals()))