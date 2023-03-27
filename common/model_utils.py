#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Model utility functions."""
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay, PiecewiseConstantDecay
from tensorflow.keras.experimental import CosineDecay
from tensorflow_model_optimization.sparsity import keras as sparsity


def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        #model.metrics_names.append(name)
        #model.metrics_tensors.append(loss)
        model.add_metric(metric, name=name, aggregation='mean')


def get_pruning_model(model, begin_step, end_step):
    import tensorflow as tf
    if tf.__version__.startswith('2'):
        # model pruning API is not supported in TF 2.0 yet
        raise Exception('model pruning is not fully supported in TF 2.x, Please switch env to TF 1.x for this feature')

    pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                   final_sparsity=0.7,
                                                   begin_step=begin_step,
                                                   end_step=end_step,
                                                   frequency=100)
    }

    pruning_model = sparsity.prune_low_magnitude(model, **pruning_params)
    return pruning_model


def get_lr_scheduler(learning_rate, decay_type, decay_steps):
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type == None:
        lr_scheduler = learning_rate
    elif decay_type == 'cosine':
        lr_scheduler = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, alpha=0.2) # use 0.2*learning_rate as final minimum learning rate
    elif decay_type == 'exponential':
        lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.9)
    elif decay_type == 'polynomial':
        lr_scheduler = PolynomialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, end_learning_rate=learning_rate/100)
    elif decay_type == 'piecewise_constant':
        #apply a piecewise constant lr scheduler, including warmup stage
        boundaries = [500, int(decay_steps*0.9), decay_steps]
        values = [0.001, learning_rate, learning_rate/10., learning_rate/100.]
        lr_scheduler = PiecewiseConstantDecay(boundaries=boundaries, values=values)
    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer(optim_type, learning_rate, average_type=None, decay_type='cosine', decay_steps=100000):
    optim_type = optim_type.lower()

    lr_scheduler = get_lr_scheduler(learning_rate, decay_type, decay_steps)

    # NOTE: you can try to use clipnorm/clipvalue to avoid run into nan, especially on new TF versions
    if optim_type == 'adam':
        optimizer = Adam(learning_rate=lr_scheduler, epsilon=1e-7, amsgrad=False, clipnorm=None, clipvalue=None)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_scheduler, rho=0.9, momentum=0.0, centered=False, clipnorm=None, clipvalue=None)
    elif optim_type == 'sgd':
        optimizer = SGD(learning_rate=lr_scheduler, momentum=0.0, nesterov=False, clipnorm=None, clipvalue=None)
    else:
        raise ValueError('Unsupported optimizer type')

    if average_type:
        optimizer = get_averaged_optimizer(average_type, optimizer)

    return optimizer


def get_averaged_optimizer(average_type, optimizer):
    """
    Apply weights average mechanism in optimizer. Need tensorflow-addons
    which request TF 2.x and have following compatibility table:
    -------------------------------------------------------------
    |    Tensorflow Addons     | Tensorflow |    Python          |
    -------------------------------------------------------------
    | tfa-nightly              | 2.3, 2.4   | 3.6, 3.7, 3.8      |
    -------------------------------------------------------------
    | tensorflow-addons-0.12.0 | 2.3, 2.4   | 3.6, 3.7, 3.8      |
    -------------------------------------------------------------
    | tensorflow-addons-0.11.2 | 2.2, 2.3   | 3.5, 3.6, 3.7, 3.8 |
    -------------------------------------------------------------
    | tensorflow-addons-0.10.0 | 2.2        | 3.5, 3.6, 3.7, 3.8 |
    -------------------------------------------------------------
    | tensorflow-addons-0.9.1  | 2.1, 2.2   | 3.5, 3.6, 3.7      |
    -------------------------------------------------------------
    | tensorflow-addons-0.8.3  | 2.1        | 3.5, 3.6, 3.7      |
    -------------------------------------------------------------
    | tensorflow-addons-0.7.1  | 2.1        | 2.7, 3.5, 3.6, 3.7 |
    -------------------------------------------------------------
    | tensorflow-addons-0.6.0  | 2.0        | 2.7, 3.5, 3.6, 3.7 |
    -------------------------------------------------------------
    """
    import tensorflow_addons as tfa

    average_type = average_type.lower()

    if average_type == None:
        averaged_optimizer = optimizer
    elif average_type == 'ema':
        averaged_optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.99)
    elif average_type == 'swa':
        averaged_optimizer = tfa.optimizers.SWA(optimizer, start_averaging=0, average_period=10)
    elif average_type == 'lookahead':
        averaged_optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=6, slow_step_size=0.5)
    else:
        raise ValueError('Unsupported average type')

    return averaged_optimizer


def get_normalize(input_shape):
    """
    rescale keypoint distance normalize coefficient
    based on input shape, used for PCK evaluation

    NOTE: 6.4 is standard normalize coefficient under
          input shape (256,256)

    # Arguments
        input_shape: input image shape as (height, width)

    # Returns
        scale: normalize coefficient
    """
    #assert input_shape[0] == input_shape[1], 'only support square input shape.'

    # use averaged scale factor for non square input shape
    scale = float((input_shape[0] + input_shape[1]) / 2) / 256.0

    return 6.4*scale
