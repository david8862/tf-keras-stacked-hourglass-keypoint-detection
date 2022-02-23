#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
from tensorflow.keras.callbacks import Callback
from hourglass.data import hourglass_dataset
from common.model_utils import get_normalize

from eval import eval_PCK


class CheckpointCleanCallBack(Callback):
    def __init__(self, checkpoint_dir, max_val_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_val_keep = max_val_keep

    def on_epoch_end(self, epoch, logs=None):

        # filter out val checkpoints
        val_checkpoints = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'ep*.h5')), reverse=False)

        # keep latest val checkpoints
        for val_checkpoint in val_checkpoints[:-(self.max_val_keep)]:
            os.remove(val_checkpoint)


class EvalCallBack(Callback):
    def __init__(self, log_dir, dataset_path, class_names, model_input_shape, model_type):
        self.log_dir = log_dir
        self.dataset_path = dataset_path
        self.class_names = class_names
        self.normalize = get_normalize(model_input_shape)
        self.model_input_shape = model_input_shape
        self.best_acc = 0.0

        self.eval_dataset = hourglass_dataset(self.dataset_path, batch_size=1, class_names=self.class_names,
                              input_shape=self.model_input_shape, num_hgstack=1, is_train=False, with_meta=True)

        # record model & dataset name to draw training curve
        with open(os.path.join(self.log_dir, 'val.txt'), 'w+') as xfile:
            xfile.write('model:' + model_type + ';dataset:' + self.eval_dataset.get_dataset_name() + '\n')
        xfile.close()

    def on_epoch_end(self, epoch, logs=None):
        val_acc, _ = eval_PCK(self.model, 'H5', self.eval_dataset, self.class_names, self.model_input_shape, score_threshold=0.5, normalize=self.normalize, conf_threshold=1e-6, save_result=False)
        print('validate accuray', val_acc, '@epoch', epoch)

        # record accuracy for every epoch to draw training curve
        with open(os.path.join(self.log_dir, 'val.txt'), 'a+') as xfile:
            xfile.write('Epoch ' + str(epoch) + ':' + str(val_acc) + '\n')
        xfile.close()

        if val_acc > self.best_acc:
            # Save best accuray value and model checkpoint
            checkpoint_dir = os.path.join(self.log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}.h5'.format(epoch=(epoch+1), loss=logs.get('loss'), val_acc=val_acc))
            self.model.save(checkpoint_dir)
            print('Epoch {epoch:03d}: val_acc improved from {best_acc:.3f} to {val_acc:.3f}, saving model to {checkpoint_dir}'.format(epoch=epoch+1, best_acc=self.best_acc, val_acc=val_acc, checkpoint_dir=checkpoint_dir))
            self.best_acc = val_acc
        else:
            print('Epoch {epoch:03d}: val_acc did not improve from {best_acc:.3f}'.format(epoch=epoch+1, best_acc=self.best_acc))
