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
from data import EEGDataUtils, EEGData
import config

import warnings
warnings.filterwarnings("ignore")

# prepare PyTorch datasets
recording_ts_labeled = EEGDataUtils.prepare_eeg_csv()

# recording_ts_labeled.to_csv('recording_ts_labeled.csv')

# initial EEG Experiment
eeg = EEG()

# prepare data loaders

# IDs and labels
partition = EEGDataUtils.prepare_partition(recording_ts_labeled, val_split=0.3)

all_labels = recording_ts_labeled['class_label']

# generators
training_set = EEGData(partition['train'], all_labels, recording_ts_labeled)
t_generator = data.DataLoader(training_set, batch_size=config.BATCH_SIZE)

validation_set = EEGData(partition['validation'], all_labels, recording_ts_labeled)
v_generator = data.DataLoader(validation_set, batch_size=config.BATCH_SIZE)

# Tensorboard writers
train_writer = SummaryWriter(config.visualization_dir + '/' + 'train')
val_writer = SummaryWriter(config.visualization_dir + '/' + 'val')

# classifier
net = nn.DataParallel(MLP())

# move model and its buffers to GPU
net.cuda()

tracker = utils.Tracker()

# optimizer
optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

for epoch in range(config.NUM_EPOCHS):
    eeg.train(model=net, data_loader=t_generator, tracker=tracker, optimizer=optimizer, writer=train_writer, epoch=epoch)
    eeg.validate(model=net, data_loader=v_generator, tracker=tracker, writer=val_writer, epoch=epoch)

