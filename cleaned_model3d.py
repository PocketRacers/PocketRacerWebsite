#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, BatchNormalization
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.callbacks import Callback, TensorBoard
from data_utils.data_processor import load_3d_dataset
from model.models import build_transformer3d
from multiprocessing import Pool
from math import sqrt
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

configproto = tf.compat.v1.ConfigProto()
configproto.gpu_options.allow_growth = True
configproto.gpu_options.polling_inactive_delay_msecs = 10
sess = tf.compat.v1.Session(config=configproto)
tf.compat.v1.keras.backend.set_session(sess)

img_size_n = 32
image_size = (img_size_n, img_size_n)

class EvaluateEndCallback(Callback):
    def on_evaluate_end(self, epoch, logs=None):
        print('epoch: {}, logs: {}'.format(epoch, logs))

class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, loss_name, file_name):
        super(PlotLosses, self).__init__()
        self.loss_name = loss_name
        self.file_name = file_name
        self.fig = plt.figure()
        self.logs = []

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get(self.loss_name))
        self.val_losses.append(logs.get('val_' + self.loss_name))
        self.i += 1
        plt.plot(self.x, self.losses, 'g')
        plt.plot(self.x, self.val_losses, 'r')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.file_name)

plot_steering_losses = PlotLosses('angle_out_loss', 'steering_loss_history_vanilla.png')
plot_throttle_losses = PlotLosses('throttle_out_loss', 'throttle_loss_history_vanilla.png')

class Generator(tf.keras.utils.Sequence):
    def __init__(self, image, steering, throttle, batch_size):
        self.image = image
        self.steering = steering
        self.throttle = throttle
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image) / float(self.batch_size))).astype(np.int32)

    def __getitem__(self, idx):
        batch_image = self.image[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_steering = self.steering[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_throttle = self.throttle[idx * self.batch_size: (idx + 1) * self.batch_size]
        return [np.stack(batch_image)], [batch_steering]

def Predict_Image_Generator(image):
    return [np.stack(image)]

def main(*args, **kwargs):
    if kwargs['n_jump'] == 0:
        kwargs['n_jump'] = kwargs['n_stacked']

    saved_weight_name = './vit3d2.h5'
    saved_file_name = './vit3d2.hdf5'.format(kwargs['n_stacked'], kwargs['n_jump'], kwargs['depth'])

    data_path = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'dataset')
    img_path = os.path.join(kwargs['img_path'])
    n_stacked = kwargs['n_stacked']

    train_images, val_images, test_images, train_steering, val_steering, test_steering, train_throttle, val_throttle, test_throttle = load_3d_dataset(
        n_stacked, img_path, h=kwargs['height'], w=kwargs['width'], d=kwargs['depth'], concatenate=kwargs['concatenate'],
        prediction_mode=kwargs['prediction_mode'], val_size=0.2, test_size=0.1, n_jump=kwargs['n_jump'])

    print("number of train images:", np.shape(train_images))
    print("number of validation images:", np.shape(val_images))
    print("number of test images:", np.shape(test_images))

    training_batch_generator = Generator(np.array(train_images[:]), np.array(train_steering[:]), np.array(train_throttle[:]), kwargs['batch_size'])
    validation_batch_generator = Generator(np.array(val_images[:]), np.array(val_steering[:]), np.array(val_throttle[:]), kwargs['batch_size'])

    directory = '/mnt/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/hallways18/single_agent_red_comet/'
    os.chdir(directory)
    with tf.device('/device:GPU:1'):
        tfboard = TensorBoard(log_dir='./logs', histogram_freq=0)
        model = build_transformer3d(kwargs['width'], kwargs['height'], kwargs['depth'], kwargs['n_stacked'])

        stop_callbacks = callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min', min_delta=0)
        checkpoint = callbacks.ModelCheckpoint(saved_weight_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

        try:
            model.load_weights('vit3d2.h5')
        except:
            print("load failed")

        model.save('vit3d2.hdf5')

        converter_lite = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_lite.experimental_new_converter = True
        converter_lite.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_lite.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model_lite = converter_lite.convert()

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        tflite_model = converter.convert()

        with open('vit3d2.tflite', 'wb') as f:
            f.write(tflite_model)

        with open('vit_lite3d2.tflite', 'wb') as f:
            f.write(tflite_model_lite)

        print("Start test....")
        model.load_weights(saved_weight_name)

        val_imgs = Predict_Image_Generator(np.array(val_images[:]))
        test_imgs = Predict_Image_Generator(np.array(test_images[:]))
        train_imgs = Predict_Image_Generator(np.array(train_images[:]))

        model_steering_val = model.predict([val_imgs], batch_size=1, verbose=0)
        model_steering_test = model.predict([test_imgs], batch_size=1, verbose=0)
        model_steering_train = model.predict([train_imgs], batch_size=1, verbose=0)

        val_imgs = []

        print("val result...")
        mae = sqrt(mean_absolute_error(val_steering[:], model_steering_val[:]))
        print('steering mae: ' + str(mae))
        rmse = sqrt(mean_squared_error(val_steering[:], model_steering_val[:]))
        print('steering rmse: ' + str(rmse))
        R2_val = r2_score(val_steering[:], model_steering_val[:])
        print('vit steering R^2: ' + str(R2_val))

        test_imgs = Predict_Image_Generator(np.array(test_images[:]))

        print("test result...")
        mae = sqrt(mean_absolute_error(test_steering[:], model_steering_test[:]))
        print('steering mae: ' + str(mae))
        rmse = sqrt(mean_squared_error(test_steering[:], model_steering_test[:]))
        print('steering rmse: ' + str(rmse))
        R2_test = r2_score(test_steering[:], model_steering_test[:])
        print('Steering R^2: ' + str(R2_test))

        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=SMALL_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

        plt.hist([np.squeeze(train_steering, axis=(1,)), np.squeeze(model_steering_train, axis=(1,))], color=['r', 'b'], bins=100)
        plt.xlim(-0.7, 0.7)
        plt.legend(['steering_target', 'steering_predict'])
        plt.xlabel('Steering Values')
        plt.savefig('steering_distribution.png')
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_stacked", help="# of stacked frame for time axis", type=int, default=2)
    parser.add_argument("--n_jump", help="time interval to get input, 0 for n_jump=n_stacked", type=int, default=1)
    parser.add_argument("--width", help="width of input images", type=int, default=img_size_n)
    parser.add_argument("--height", help="height of input images", type=int, default=img_size_n)
    parser.add_argument("--depth", help="the number of channels of input images", type=int, default=6)
    parser.add_argument("--img_path", help="image directory", type=str, default='/mnt/c334c9bc-7ae4-4ea7-84fb-6b8f5595aea2/hallways18/single_agent_red_comet/data')
    parser.add_argument("--epochs", help="total number of training epochs", type=int, default=50000)
    parser.add_argument("--batch_size", help="batch_size", type=int, default=12000)
    parser.add_argument("--concatenate", help="target csv filename", type=int, default=1)
    parser.add_argument("--prediction_mode", help="sets steering predictions from categorical cross entropy or from linear regression", type=str, default='linear')

    args = parser.parse_args()
    main(**vars(args))
