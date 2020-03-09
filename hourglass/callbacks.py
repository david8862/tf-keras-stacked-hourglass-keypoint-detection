#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback
from hourglass.postprocess import post_process_heatmap
from hourglass.data import hourglass_dataset
from common.model_utils import get_normalize
from eval import eval_PCK


class EvalCallBack(Callback):
    def __init__(self, log_dir, dataset_path, class_names, input_size, model_type):
        self.log_dir = log_dir
        self.dataset_path = dataset_path
        self.class_names = class_names
        self.normalize = get_normalize(input_size)
        self.input_size = input_size
        self.best_accuray = 0.0

        val_dataset = hourglass_dataset(self.dataset_path, self.class_names,
                              input_size=self.input_size, is_train=False)

        # record model & dataset name to draw training curve
        with open(os.path.join(self.log_dir, 'val.txt'), 'a+') as xfile:
            xfile.write('model:' + model_type + ';dataset:' + val_dataset.get_dataset_name() + '\n')
        xfile.close()

    def on_epoch_end(self, epoch, logs=None):
        val_dataset = hourglass_dataset(self.dataset_path, self.class_names,
                              input_size=self.input_size, is_train=False)

        val_acc, _ = eval_PCK(self.model, 'H5', val_dataset, self.class_names, score_threshold=0.5, normalize=self.normalize, conf_threshold=1e-6, save_result=False)
        print('validate accuray', val_acc, '@epoch', epoch)

        # record accuracy for every epoch to draw training curve
        with open(os.path.join(self.log_dir, 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(val_acc) + '\n')
        xfile.close()

        if val_acc > self.best_accuray:
            # Save best accuray value and model checkpoint
            self.best_accuray = val_acc
            model_file = 'ep{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}.h5'.format(epoch=(epoch+1), loss=logs.get('loss'), val_acc=val_acc)
            print("Saving model to", model_file)
            self.model.save(os.path.join(self.log_dir, model_file))
