import math

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')

def rnn_model_factory_bayesian(
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

def train_step_generator_bayesian(model, optimizer, nan_value=0):
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