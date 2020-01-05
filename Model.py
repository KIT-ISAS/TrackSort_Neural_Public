import math
import tensorflow as tf
import numpy as np
import code # code.interact(local=dict(globals(), **locals()))

from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')

rnn_models = {
    'lstm': tf.keras.layers.LSTM,
    'rnn': tf.keras.layers.SimpleRNN,
    'gru': tf.keras.layers.GRU
}


def rnn_model_factory(
        num_units_first_rnn=16, num_units_second_rnn=16, num_units_third_rnn=0, num_units_fourth_rnn=0,
        num_units_first_dense=16, num_units_second_dense=0, num_units_third_dense=0, num_units_fourth_dense=0,
        rnn_model_name='lstm',
        use_batchnorm_on_dense=True,
        num_time_steps=35, batch_size=128, nan_value=0, input_dim=2, unroll=True, stateful=True):
    """
    Create a new keras model with the sequential API

    Note: The number of layers is fixed, because the HParams Board can't handle lists at the moment (Nov-2019)

    :param num_units_first_rnn:
    :param num_units_second_rnn:
    :param num_units_third_rnn:
    :param num_units_fourth_rnn:
    :param num_units_first_dense:
    :param num_units_second_dense:
    :param num_units_third_dense:
    :param num_units_fourth_dense:
    :param rnn_model_name:
    :param use_batchnorm_on_dense:
    :param num_time_steps:
    :param batch_size:
    :param nan_value:
    :param input_dim:
    :param unroll:
    :param stateful: If this is False, then the state of the rnn will be reset after every batch.
            We want to control this manually therefore the default is True.
    :return: the model and a hash string identifying the architecture uniquely
    """
    model = tf.keras.Sequential()

    # hash for the model architecture
    hash_ = 'v1'

    # The Masking makes the model ignore time steps where the whole vector
    # consists of the *mask_value*
    model.add(tf.keras.layers.Masking(mask_value=nan_value, name="masking_layer",
                                      batch_input_shape=(batch_size, num_time_steps, input_dim)))
    hash_ += "-masking"

    # get the rnn layer
    rnn_model = rnn_models[rnn_model_name]

    # Add the recurrent layers
    rnn_layer_count = 0
    for hidden_units_in_rnn in [num_units_first_rnn, num_units_second_rnn, num_units_third_rnn, num_units_fourth_rnn]:
        if hidden_units_in_rnn == 0:
            # as soon as one layer has no units, we don't create the layer.
            break
        else:
            hash_ += "-{}[{}]".format(rnn_model_name, hidden_units_in_rnn)
            model.add(rnn_model(hidden_units_in_rnn,
                                return_sequences=True,
                                stateful=stateful,
                                name='rnn-{}'.format(rnn_layer_count),
                                recurrent_initializer='glorot_uniform',
                                unroll=unroll))
            rnn_layer_count += 1

    # Add the dense layers
    for units_in_dense_layer in [num_units_first_dense, num_units_second_dense, num_units_third_dense,
                                 num_units_fourth_dense]:
        if units_in_dense_layer == 0:
            # as soon as one layer has no units, we don't create the layer, neither further ones
            break
        else:
            hash_ += "-dense[{}, leakyrelu]".format(units_in_dense_layer)
            model.add(tf.keras.layers.Dense(units_in_dense_layer))
            model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
            if use_batchnorm_on_dense:
                model.add(tf.keras.layers.BatchNormalization())
                hash_ += "-BatchNorm"

    # Always end with a dense layer with two outputs (x, y)
    model.add(tf.keras.layers.Dense(2))

    hash_ += "-dense[2]"

    return model, hash_


def tf_error(model, dataset, normalization_factor, squared=True, nan_value=0):
    """
    Build a function which returns a computational graph for tensorflow.

    :param model: keras model api
    :param dataset: iterable with shape [batch_size, time_steps, num_dims*2]
    :param normalization_factor: all floats get multiplied by this (usually belt_width)
    :param squared: if true, return MSE. Else: MAE
    :param nan_value: the value which was used for padding (usually 0)

    Example:
        >>> import data
        >>> keras_model = rnn_model_factory()
        >>> _, dataset_test = data.FakeDataSet().get_tf_data_sets_seq2seq_data()
        >>> calc_mse_test = model.tf_error(keras_model, dataset_test, normalization_factor, squared=True)
        >>> mse = calc_mse_test()
        >>> print("MSE: {}".format(mse))

    :return: method that constructs the graph. The result of the method can be called to calc the error
    """
    # the placeholder character used for padding
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)
    norm_factor = tf.constant(normalization_factor, dtype=tf.float64)

    @tf.function
    def f():
        # AutoGraph has special support for safely converting for loops when y is a tensor or tf.data.Dataset.
        # https://www.tensorflow.org/tutorials/customization/performance

        loss = tf.constant(0, dtype=tf.float64)
        step_counter = tf.constant(0, dtype=tf.float64)

        for input_batch, target_batch in dataset:
            # reset state
            hidden = model.reset_states()

            batch_predictions = model(input_batch)

            # Calculate the mask
            mask = K.all(K.equal(target_batch, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            # Revert the normalization
            # ToDo: Replace this hacky solution with same normalization in both directions
            #    with usage of x_max and y_max   with tf.scatter_nd_update(...)
            target_batch_unnormalized = target_batch * norm_factor
            pred_batch_unnormalized = batch_predictions * norm_factor

            if squared:
                batch_loss = tf.keras.losses.mean_squared_error(target_batch_unnormalized,
                                                                pred_batch_unnormalized) * mask
            else:
                batch_loss = tf.keras.losses.mean_absolute_error(target_batch_unnormalized,
                                                                 pred_batch_unnormalized) * mask

            # take average w.r.t. the number of unmasked entries
            loss += K.sum(batch_loss)
            step_counter += K.sum(mask)

        return loss/step_counter

    return f


def train_step_generator(model, optimizer, nan_value=0):
    """
    Build a function which returns a computational graph for tensorflow.
    This function can be called to train the given model with the given
    optimizer.

    :param model: model according to estimator api
    :param optimizer: tf estimator
    :param nan_value: e.g. 0

    Example:
        >>> import data
        >>> keras_model = rnn_model_factory()
        >>> dataset_train, _ = data.FakeDataSet().get_tf_data_sets_seq2seq_data()
        >>> train_step = model.train_step_generator(model, optimizer)
        >>>
        >>> step, batch_size = 0, 32
        >>> for epoch in range(100):
        >>>     for (batch_n, (inp, target)) in enumerate(dataset_train):
        >>>         # _ = keras_model.reset_states()
        >>>         loss = train_step(inp, target)
        >>>         step += batch_size
        >>>     tf.summary.scalar('loss', loss, step=step)

    :return: function which can be called to train the given model with the given optimizer
    """
    # the placeholder character used for padding
    mask_value = K.variable(np.array([nan_value, nan_value]), dtype=tf.float64)

    @tf.function
    def train_step(inp, target):
        with tf.GradientTape() as tape:
            target = K.cast(target, tf.float64)
            predictions = model(inp)

            mask = K.all(K.equal(target, mask_value), axis=-1)
            mask = 1 - K.cast(mask, tf.float64)
            mask = K.cast(mask, tf.float64)

            # multiply categorical_crossentropy with the mask
            loss = tf.keras.losses.mean_squared_error(target, predictions) * mask

            # take average w.r.t. the number of unmasked entries
            loss = K.sum(loss) / K.sum(mask)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return loss

    return train_step


def train_epoch_generator(rnn_model, train_step, dataset_train, batch_size):
    """
    Build a function which returns a computational graph for tensorflow.
    This function can be called to train the given model with the given
    optimizer **for a number of epochs**.
    It optimizes the training as it lets tensorflow train whole epochs
    without coming back to the python code.

    This uses the train_step. You can call the train_step function on
    your own if you want more control.

    Attention: Keyboard interrupts are not stopping the training.

    Example:
        >>> import data
        >>> dataset_train, _ = data.FakeDataSet().get_tf_data_sets_seq2seq_data()
        >>> train_step = rnn_model.train_step_generator(rnn_model,  tf.keras.optimizers.Adam())
        >>> step, batch_size = 0, 32
        >>> for epoch in range(100):
        >>>     avg_loss, train_step_counter = train_epoch(1)
        >>>     tf.summary.scalar('loss', avg_loss, step=train_step_counter)

    :param rnn_model:
    :param train_step:
    :param dataset_train:
    :param batch_size:
    :return:
    """

    @tf.function
    def train_epoch(n_epochs):
        train_step_counter = tf.constant(0, dtype=tf.float64)
        batch_counter = tf.constant(0, dtype=tf.float64)
        sum_loss = tf.constant(0, dtype=tf.float64)

        for inp, target in dataset_train.repeat(n_epochs):
            hidden = rnn_model.reset_states()
            loss = train_step(inp, target)
            sum_loss += loss
            train_step_counter += batch_size
            batch_counter += 1

        avg_loss = sum_loss/batch_counter

        return avg_loss, train_step_counter

    return train_epoch

def set_state(rnn_model, batch_state):
    rnn_layer_counter = 0
    for i in range(1000):
        try:
            layer = rnn_model.get_layer(index=i)
        except:
            break

        if isinstance(layer, tf.keras.layers.RNN):
            for sub_state_number, sub_state in enumerate(layer.states):
                layer.states[sub_state_number].assign(tf.convert_to_tensor(batch_state[rnn_layer_counter][sub_state_number]))
            rnn_layer_counter += 1


def get_state(rnn_model):
    rnn_layer_states = []
    # get all layers in ascending order
    # ToDo: Replace this with while-true with break condition
    for i in range(1000):
        #print('layer_number: ' + str(i))
        # Not asking for permission but handling the error is faster in python
        try:
            layer = rnn_model.get_layer(index=i)
        except:
            #print('layer ' + str(i) + ' does not exist')
            break

        # only store the state of the layer if it is a recurrent layer
        #   DenseLayers don't have a state
        if isinstance(layer, tf.keras.layers.RNN):
            #print('in rnn state')
            #code.interact(local=dict(globals(), **locals()))
            rnn_layer_states.append([sub_state.numpy() for sub_state in layer.states])
            # print(rnn_layer_states)

    return rnn_layer_states


class Model(object):
    def __init__(self, global_config, data_source):
        # TODO one might want to encode the hyperparams for the RNN in the global config
        self.global_config = global_config
        self.data_source = data_source
        if self.global_config['is_loaded']:
            self.rnn_model = tf.keras.models.load_model(self.global_config['model_path'])
            # self.rnn_model.load_weights('weights_path')
        else:
            self.rnn_model = rnn_model_factory()[0]
            print(self.rnn_model.summary())
            self.train()


    def get_zero_state(self):
        self.rnn_model.reset_states()
        return get_state(self.rnn_model)


    # expected to return list<vector<pair<float,float>>>, list<RNNStateTuple>
    def predict(self, current_input, state):
        new_states = []
        predictions = []
        current_input = np.expand_dims(current_input, axis=1)
        #print('in predict')
        #code.interact(local=dict(globals(), **locals()))
        set_state(self.rnn_model, state)
        prediction = self.rnn_model(current_input)
        prediction = np.squeeze(prediction)
        new_state = get_state(self.rnn_model)
        return prediction, new_state



    def train(self):
        dataset_train, _ = self.data_source.get_tf_data_sets_seq2seq_data(normalized=True)

        optimizer = tf.keras.optimizers.Adam()
        train_step_fn = train_step_generator(self.rnn_model, optimizer)


        # Train model
        for epoch in range(self.global_config['num_train_epochs']):
            # learning rate decay after 100 epochs
            if (epoch+1) % 150 == 0:
                old_lr = K.get_value(optimizer.lr)
                new_lr = old_lr * 0.1
                print("Reducing learning rate from {} to {}.".format(old_lr, new_lr))
                K.set_value(optimizer.lr, new_lr)

            for (batch_n, (inp, target)) in enumerate(dataset_train):
                _ = self.rnn_model.reset_states()
                loss = train_step_fn(inp, target)    

            print("{}/{}: \t loss={}".format(epoch, self.global_config['num_train_epochs'], loss))

        self.rnn_model.save_weights(self.global_config['weights_path'])
        self.rnn_model.save(self.global_config['model_path'])



    def predict_final(self, states, y_targetline):
        raise NotImplementedError



    def train_final(self, x_data, y_data, pred_point):
        raise NotImplementedError