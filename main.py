import torch.optim as optim
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
from models.mlp import MLP
from models.eegnet import EEGNet
from eeg import EEG
import utils as utils
from data import EEGDataUtils, EEGMlpData, EEGNetData
import config
import argparse

import warnings
warnings.filterwarnings("ignore")

def initiate_train_and_validation(args):
        
    # prepare EEG dataset
    eeg_ts_labeled = EEGDataUtils.prepare_eeg_csv()

    # Optional: save to CSV
    # eeg_ts_labeled.to_csv('eeg_ts_labeled.csv')

    # initial EEG Experiment
    eeg = EEG()

    # prepare data loaders
    if args.eegnet == True:
        # EEGNet Processing (Randomized inputs)
        # generators
        print("Preparing random data...")
        training_set = EEGNetData(prefix='train')
        t_generator = data.DataLoader(training_set, batch_size=config.BATCH_SIZE)
        validation_set = EEGNetData(prefix='validation')
        v_generator = data.DataLoader(validation_set, batch_size=config.BATCH_SIZE)
    else:
        # MLP Processing
        # IDs and labels
        print("Preparing EEG recordings...")
        partition = EEGDataUtils.prepare_partition(eeg_ts_labeled, val_split=0.3)
        all_labels = eeg_ts_labeled['class_label']
        # generators
        training_set = EEGMlpData(partition['train'], all_labels, eeg_ts_labeled)
        t_generator = data.DataLoader(training_set, batch_size=config.BATCH_SIZE)
        validation_set = EEGMlpData(partition['validation'], all_labels, eeg_ts_labeled)
        v_generator = data.DataLoader(validation_set, batch_size=config.BATCH_SIZE)

    # Tensorboard writers
    train_writer = SummaryWriter(config.visualization_dir + '/' + 'train')
    val_writer = SummaryWriter(config.visualization_dir + '/' + 'val')

    # classifier
    net = nn.DataParallel(EEGNet())

    # move model and its buffers to GPU
    net.cuda()

    tracker = utils.Tracker()

    # optimizer
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    for epoch in range(config.NUM_EPOCHS):
        eeg.train(model=net, data_loader=t_generator, tracker=tracker, optimizer=optimizer, writer=train_writer, epoch=epoch)
        eeg.validate(model=net, data_loader=v_generator, tracker=tracker, writer=val_writer, epoch=epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--eegnet', type=bool, help='Run EEGNet model with randomly generated data')
    parser.add_argument('--mlp', type=bool, help='Run MLP model with EEG recordings')

    args = parser.parse_args()

    # init train and val with given params
    initiate_train_and_validation(args)



