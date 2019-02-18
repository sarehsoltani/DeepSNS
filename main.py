import torch.optim as optim
from torch.utils import data
from mne import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from eeg import EEG
from eeg_preprocessing import EEGPreprocessing as EEGPrep
from model import MLP
from data import EEGData
import config

import warnings
warnings.filterwarnings("ignore")

root = './data'
classes = ['img_lhand', 'img_rhand', 'img_bfeet', 'img_tongue', 'img_none']

class_mapping = {
    'img_lhand': 769, 
    'img_rhand': 770,
    'img_bfeet': 771,
    'img_tongue': 772,
}

# load recording gdf and extract its headers
bci_record_names = []
for file_name in os.listdir(root):
    bci_record_names.append(file_name)

# stim channel is required for gdf files
gdf_file = io.read_raw_edf(root + '/' + bci_record_names[3], stim_channel=1, preload=True)
recording_headers = io.find_edf_events(gdf_file)
recording_positions = recording_headers[1]
recording_types = recording_headers[2]
recording_durations = recording_headers[4]

# convert to dataframe (recording time series)
recording_ts = gdf_file.to_data_frame()

# extract class ranges
class_ranges = EEGPrep.get_class_ranges(class_mapping, recording_types, recording_positions, recording_durations)

# add a new column for labels
recording_ts_labeled = EEGPrep.label_recording(recording_ts, class_ranges)

# fill all nans with 'none'
recording_ts_labeled['class_label'].fillna('img_none', inplace=True)
recording_ts_labeled['class_label'].replace(0.0, 'img_none', inplace=True)

# prepare PyTorch datasets

# initial EEG Experiment
eeg = EEG()

# prepare data loaders

# IDs and labels
partition = {}
labels = []

# generators
training_set = EEGData(partition['train'], labels)
t_generator = data.DataLoader(training_set, batch_size=config.BATCH_SIZE)

validation_set = EEGData(partition['validation'], labels)
v_generator = data.DataLoader(validation_set, batch_size=config.BATCH_SIZE)
# classifier
net = MLP()

# optimizer
optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

for epoch in range(config.NUM_EPOCHS):
    eeg.train(model=net, data_loader=loader, optimizer=optimizer)
    eeg.evaluate(model=net, data_loader=loader)
