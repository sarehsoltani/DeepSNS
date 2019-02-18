import torch.optim as optim
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
from model import MLP
from eeg import EEG
from data import EEGData, EEGDataUtils
import config

import warnings
warnings.filterwarnings("ignore")

# prepare PyTorch datasets
recording_ts_labeled = EEGDataUtils.prepare_eeg_csv()

# initial EEG Experiment
eeg = EEG()

# prepare data loaders

# IDs and labels
partition = EEGDataUtils.prepare_partition(recording_ts_labeled, val_split=0)

all_labels = recording_ts_labeled['class_label']

# generators
training_set = EEGData(partition['train'], all_labels)
t_generator = data.DataLoader(training_set, batch_size=config.BATCH_SIZE)

validation_set = EEGData(partition['validation'], all_labels)
v_generator = data.DataLoader(validation_set, batch_size=config.BATCH_SIZE)

# classifier
net = MLP()

# optimizer
optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

for epoch in range(config.NUM_EPOCHS):
    eeg.train(model=net, data_loader=t_generator, optimizer=optimizer, epoch=epoch)
    eeg.evaluate(model=net, data_loader=v_generator, epoch=epoch)

