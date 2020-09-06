#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_accuracy(data_path):
    """
    Load & parse the accuracy data file
      accuracy data format:

           model:hg_s2_mobile_tiny_192_192;dataset:MPI
           Epoch 0:0.03907726641313506
           Epoch 1:0.3704370004326737
           ...

      return:
           label:    training model + dataset, used for
                     plotting curve
           epoch:    list of epoch numbers
           accuracy: list of accuracy for every epoch
    """
    with open(data_path) as f:
        acc_data = f.readlines()
    acc_data = [c.strip() for c in acc_data]

    # parse line 1 to get model & dataset name for curve label
    model = acc_data[0].split(';')[0].split(':')[-1]
    dataset = acc_data[0].split(';')[-1].split(':')[-1]
    label = model + '_' + dataset

    # parse accuracy data
    accuracy = [float(c.split(':')[-1]) for c in acc_data[1:]]
    epoch = [int(c.split(':')[0].split(' ')[-1]) for c in acc_data[1:]]

    return label, epoch, accuracy


def set_plot_color(num_plots):
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, num_plots)])


def main(args):
    file_list = args.data_files.split(',')
    # set color pattern for different curves
    set_plot_color(len(file_list))

    for i, data_file in enumerate(file_list):
        # parse and draw val accuracy curve
        label, epoch, accuracy = get_accuracy(data_file)
        plt.plot(epoch, accuracy, label=label)

    # set chart title & label
    plt.title('val accuracy', fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.ylabel('accuracy', fontsize='large')
    plt.legend(loc='lower right')

    # save chart image & show it
    plt.savefig(args.output_file, dpi=75)
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw val accuracy curves during training')

    parser.add_argument('--data_files', type=str, required=True, help='val accuracy record files, separated with comma')
    parser.add_argument('--output_file', type=str, required=False, help='saved curve chart image file, default=%(default)s', default='./accuracy.jpg')

    args = parser.parse_args()

    main(args)
