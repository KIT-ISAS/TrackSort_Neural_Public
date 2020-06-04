import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
import matplotlib

plt = matplotlib.pyplot
tfd = tfp.distributions

def test_separation_mlp_advanced_training(data_manager):
    time_normalization=15.71
    normalization_constant = data_manager.normalization_constant
    mlp_dataset_train, mlp_dataset_eval, mlp_dataset_test = \
            data_manager.get_tf_data_sets_mlp_with_separation_data( 
                normalized=True, 
                evaluation_ratio = 0.15, 
                test_ratio= 0.15,
                batch_size=100, 
                random_seed=0,
                time_normalization=time_normalization,
                n_inp_points = 7)

    seq2seq_dataset_train, seq2seq_dataset_eval, seq2seq_dataset_test, num_time_steps = \
        data_manager.get_tf_data_sets_seq2seq_with_separation_data(
            normalized=True, 
            evaluation_ratio = 0.15, 
            test_ratio= 0.15,
            batch_size=100, 
            random_seed=0,
            time_normalization=time_normalization)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(2),])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.005,
        decay_steps=200,
        decay_rate=0.96,
        staircase=True)

    n_epochs = 1000

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss=tf.keras.losses.MeanSquaredError()
    train_step_fn = train_step_generator_separation_prediction(model, optimizer, loss)

    ## Start training
    train = True
    if train:
        for i in range(n_epochs):
            losses = []
            seq2seq_iter = iter(seq2seq_dataset_train)
            mlp_iter = iter(mlp_dataset_train)
            for (seq2seq_inp, seq2seq_target, seq2seq_tracking_mask, seq2seq_separation_mask) in seq2seq_iter:
                (mlp_inp, mlp_target, mlp_mask) = next(mlp_iter)
                train_loss = train_step_fn(mlp_inp, mlp_target, mlp_mask, True)
                losses.append(train_loss.numpy())
            print("Finished epoch {}/{} with mean loss: {}".format(i, n_epochs, sum(losses)/len(losses)))
        
        ## Get eval data
        model.save("models/pepper/MLP_separator_fitting_test.h5")
    else:
        model = tf.keras.models.load_model("models/pepper/MLP_separator_fitting_test.h5")
    all_inputs = np.array([]); all_targets = np.array([]); all_predictions = np.array([]); all_masks = np.array([]); all_weights = np.array([])
    
    seq2seq_iter = iter(seq2seq_dataset_eval)
    mlp_iter = iter(mlp_dataset_eval)
    for (seq2seq_inp, seq2seq_target, seq2seq_tracking_mask, seq2seq_separation_mask) in seq2seq_iter:
        (mlp_inp, mlp_target, mlp_mask) = next(mlp_iter)
        # Test experts on a batch
        predictions = []
        model_prediction = model(mlp_inp)
        predictions.append(model_prediction)
        masks = [mlp_mask]
        # Convert all masks and predictions to MLP style
        np_predictions = []
        np_masks = []
        for i in range(len(predictions)):
            if tf.is_tensor(predictions[i]):
                np_prediction = predictions[i].numpy()
            else:
                np_prediction = predictions[i]
            # If seq2seq
            if np_prediction.ndim == 3:
                np_prediction = np_prediction[np.where(masks[i].numpy())][:,2:]
            np_masks.append(mlp_mask.numpy())
            np_predictions.append(np_prediction)
            stop=0
        # Get weighting of experts
        weights = np.zeros([len(predictions), predictions[0].shape[0]])
        
        # Add everything to the lists
        if all_targets.shape[0]==0:
            all_inputs = mlp_inp.numpy()
            all_targets = mlp_target.numpy()
            all_predictions = np.array(np_predictions)
            all_masks = np.array(np_masks)
            all_weights = weights
        else:
            all_inputs = np.concatenate((all_inputs, mlp_inp.numpy()),axis=0)
            all_targets = np.concatenate((all_targets, mlp_target.numpy()),axis=0)
            all_predictions = np.concatenate((all_predictions, np.array(np_predictions)), axis=1)
            all_masks = np.concatenate((all_masks, np.array(np_masks)), axis=1)
            all_weights = np.concatenate((all_weights, weights), axis=1)
    # Create expert name list
    expert_names = ["MLP"]
    ## Evaluation
    
    errors = all_predictions - np.repeat(all_targets[np.newaxis,:,0:2], all_predictions.shape[0], axis=0)
    spatial_errors = errors[:,:,0] * normalization_constant
    temporal_errors = errors[:,:,1] * time_normalization
    # Create box values for every expert
    spatial_boxplot_inputs = []; temporal_boxplot_inputs = []
    for i in range(spatial_errors.shape[0]):
        # Spatial error
        spatial_error = spatial_errors[i, np.where(masks[i])]
        spatial_boxplot_inputs.append(spatial_error[0])
        # Temporal error
        temporal_error = temporal_errors[i, np.where(masks[i])]
        temporal_boxplot_inputs.append(temporal_error[0])

    # Show temporal plot
    plt.figure()
    plt.boxplot(temporal_boxplot_inputs, sym='', labels=expert_names)
    plt.ylabel("Temporal deviation [frames]")
    plt.ylim([-1, 1])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=60)
    plt.show()
    # Show spatial plot
    plt.figure()
    plt.boxplot(spatial_boxplot_inputs, sym='', labels=expert_names)
    plt.ylabel("Spatial deviation [px]")
    plt.ylim([-50, 50])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=60)
    plt.show()  
    """
    ## Evaluation
    train_data, eval_data = get_np_dataset_from_datamanager(data_manager)
    x_eval = eval_data[:, :-4]
    y_eval_t = eval_data[:,-3]
    y_eval_y = eval_data[:,-4]
    y_eval=eval_data[:,-4:-2]
    yhat = model(x_eval)
    
    error = (yhat.numpy()[:,1]-y_eval[:,1]) * time_normalization 
    plt.figure()
    plt.boxplot(error, sym='', labels=['TFP MLP'])
    plt.ylabel("Temporal deviation [frames]")
    if normalization_constant >= 100:
        plt.ylim([-1, 1])
    else:
        plt.ylim([-4, 4])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=60)
    plt.show()
    # Show spatial plot
    error = (yhat.numpy()[:,0]-y_eval[:,0]) * normalization_constant #.mean()
    #error = (yhat.numpy()[:,1]-y_eval[:,1]) * normalization_constant
    plt.figure()
    plt.boxplot(error, sym='', labels=['TFP MLP'])
    plt.ylabel("Spatial deviation [px]")
    plt.ylim([-50, 50])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=60)
    plt.show()
    """
    stop=0
        
                

def train_step_generator_separation_prediction(model, optimizer, loss_object):
    @tf.function
    def train_step(inp, target, mask, training=True):
        with tf.GradientTape() as tape:
            predictions = model(inp, training=training, mask=mask)
            loss = loss_object(tf.gather(target, [0,1], axis=1), predictions, sample_weight=mask)
        if training:
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss
    return train_step

def test_separation_mlp(data_manager):
    time_normalization=15.71
    normalization_constant = data_manager.normalization_constant
    train_data, eval_data = get_np_dataset_from_datamanager(data_manager)
    x = train_data[:, :-4]
    y_t = train_data[:,-3]
    y_y = train_data[:,-4]
    y=train_data[:,-4:-2]
    input_shape = x.shape[1]
    # Build model.
    # Version 1: Only predict mean
    """
    model = tf.keras.Sequential([
        #tf.keras.layers.Dense(input_shape),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(1),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
    ])
    """
    # Version 2: Predict scale as well
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(2),
        #tfp.layers.DistributionLambda(
        #    lambda t: tfd.Normal(loc=t[..., :1],
        #                        scale=0 + tf.math.softplus(1e-5 * t[..., 1:]))),
    ])
    
    # Version 3: Epistemic uncertainty
    """
    model = tf.keras.Sequential([
        tfp.layers.DenseVariational(16, posterior_mean_field, prior_trainable),
        tfp.layers.DenseVariational(16, posterior_mean_field, prior_trainable),
        tfp.layers.DenseVariational(2, posterior_mean_field, prior_trainable),
        tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
                                scale=0 + tf.math.softplus(1e-5 * t[..., 1:]))),
        ])
    """
    # Do inference.
    negloglik = lambda y, rv_y: -rv_y.log_prob(y)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.005,
        decay_steps=200,
        decay_rate=0.96,
        staircase=True)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.MeanSquaredError())#loss=negloglik) 
    model.fit(x, y, epochs=1000, batch_size=100, verbose=True)

    # Profit.
    #[print(np.squeeze(w.numpy())) for w in model.weights]
    x_eval = eval_data[:, :-4]
    y_eval_t = eval_data[:,-3]
    y_eval_y = eval_data[:,-4]
    y_eval=eval_data[:,-4:-2]
    yhat = model(x_eval)
    #std_dev = yhat.stddev()

    ## Plots
    
    error = (yhat.numpy()[:,1]-y_eval[:,1]) * time_normalization #.mean()
    plt.figure()
    plt.boxplot(error, sym='', labels=['TFP MLP'])
    plt.ylabel("Temporal deviation [frames]")
    if normalization_constant >= 100:
        plt.ylim([-1, 1])
    else:
        plt.ylim([-4, 4])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=60)
    plt.show()
    """
    plt.figure()
    plt.scatter(np.abs(error), std_dev)
    plt.xlabel('|Temporal Error|')
    plt.ylabel('Predicted std dev')
    plt.show()"""
    # Show spatial plot
    error = (yhat.numpy()[:,0]-y_eval[:,0]) * normalization_constant #.mean()
    #error = (yhat.numpy()[:,1]-y_eval[:,1]) * normalization_constant
    plt.figure()
    plt.boxplot(error, sym='', labels=['TFP MLP'])
    plt.ylabel("Spatial deviation [px]")
    plt.ylim([-50, 50])
    plt.grid(b=True, which='major', axis='y', linestyle='--')
    plt.xticks(rotation=60)
    plt.show()
    """
    plt.figure()
    plt.scatter(np.abs(error), std_dev)
    plt.xlabel('|Spatial Error|')
    plt.ylabel('Predicted std dev')
    plt.show()
    """
    #model.save("models/pepper/MLP_separator_fitting_test.h5")
    stop=0

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return tf.keras.Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])

def get_np_dataset_from_datamanager(data_manager):
    normalized=True
    evaluation_ratio=0.15
    test_ratio=0.15
    batch_size=64
    random_seed=0
    time_normalization=15.71
    n_inp_points = 7

    track_data = data_manager.separation_track_data
    spatial_labels = data_manager.separation_spatial_labels
    temporal_labels = data_manager.separation_temporal_labels
    y_velocity_labels = data_manager.y_velocity_labels

    n_tracks = track_data.shape[0]
    n_inp_points = 7
    mlp_data = np.zeros([n_tracks, 2*n_inp_points + 4])
    # Build MLP input and target for all tracks
    missing_track_counter = 0
    for i in range(n_tracks):
        n_measurements = data_manager.get_last_timestep_of_track(track_data[i])
        if n_measurements >= n_inp_points:
            # Set input = [x1, x2, ..., y_1, y2, ...]
            c=0
            for index in np.arange(n_measurements-n_inp_points, n_measurements):
                mlp_data[i,c] = track_data[i, index, 0]
                mlp_data[i,c+n_inp_points] = track_data[i, index, 1]
                c += 1
            # Set target [y_nozzle, dt_nozzle]
            mlp_data[i, -4] = spatial_labels[i, 1]
            mlp_data[i, -3] = temporal_labels[i]
            mlp_data[i, -2] = y_velocity_labels[i]
            mlp_data[i, -1] = 1
        else:
            missing_track_counter += 1
            mlp_data[i, -1] = 0

    #logging.info("Skipping {} tracks in the MLP separation prediction because their length was < {}".format(missing_track_counter, n_inp_points))

    if normalized:
        # normalize spatial data
        mlp_data[:, :-3] /= data_manager.normalization_constant
        # normalize temporal prediction
        mlp_data[:, -3] /= time_normalization
        # normalize velocity prediction
        mlp_data[:, -2] /= data_manager.normalization_constant

    mlp_data = mlp_data[mlp_data[:,-1]==1]

    t_e_data, test_data = train_test_split(mlp_data, test_size=test_ratio, random_state=0)
    train_data, eval_data = train_test_split(t_e_data, test_size=evaluation_ratio, random_state=0)


    
    return train_data, eval_data