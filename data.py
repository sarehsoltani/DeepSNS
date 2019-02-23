import torch
from torch.utils.data import Dataset
from mne import io
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from eeg import EEG
from eeg_preprocessing import EEGPreprocessing as EEGPrep
import os


class EEGDataUtils:

    def __init__(self):
        pass

    @staticmethod
    def one_hot(df, cols):
        """
        @param df pandas DataFrame
        @param cols a list of columns to encode 
        @return a DataFrame with one-hot encoding
        """
        for each in cols:
            dummies = pd.get_dummies(df[each])
            df = pd.concat([df, dummies], axis=1)

        return df

    @staticmethod
    def prepare_partition(eeg_csv, val_split=0.7):
        
        partition = {}

        train, val = train_test_split(eeg_csv, test_size=val_split)

        partition['train'] = train.index.values
        partition['validation'] = val.index.values

        return partition

    @staticmethod
    def prepare_eeg_csv():

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

        print('Filename: {}'.format(bci_record_names[0]))
        # stim channel is required for gdf files
        gdf_file = io.read_raw_edf(root + '/' + bci_record_names[0], stim_channel=1, preload=True)
        recording_headers = io.find_edf_events(gdf_file)
        recording_positions = recording_headers[1]
        recording_types = recording_headers[2]
        recording_durations = recording_headers[4]

        # convert to dataframe (recording time series)
        recording_ts = gdf_file.to_data_frame()

        # fill nans with 0
        recording_ts.fillna(0.0, inplace=True)

        # normalize
        recording_ts = (recording_ts - recording_ts.min()) / recording_ts.std()

        # extract class ranges
        class_ranges = EEGPrep.get_class_ranges(class_mapping, recording_types, recording_positions, recording_durations)

        # add a new column for labels
        recording_ts_labeled = EEGPrep.label_recording(recording_ts, class_ranges)

        # fill all nans with 'none'
        recording_ts_labeled['class_label'].fillna('img_none', inplace=True)
        recording_ts_labeled['class_label'].replace(0.0, 'img_none', inplace=True)
        
        # convert to one-hot
        recording_ts_oh = EEGDataUtils.one_hot(recording_ts_labeled, ['class_label'])

        print(recording_ts_oh.columns)

        return recording_ts_oh

    @staticmethod
    def pd_to_np(df, cols_to_maintain):

        nped = []

        if cols_to_maintain == 'all':

            for idx, row in df.iterrows():
                nped.append(row)

        else:
            for idx, row in df.iterrows():
                nped.append(row[cols_to_maintain])
        
        return np.asarray(nped, dtype=np.float32)



class EEGData(Dataset):

    classes = ['img_lhand', 'img_rhand', 'img_bfeet', 'img_tongue', 'img_none']

    def __init__(self, list_ids, labels, recording_ts_labeled):
        self.list_ids = list_ids
        self.labels = labels

        # drop labels
        self.recording_ts = recording_ts_labeled.drop('class_label', axis=1)
        
        # drop classes
        self.feature_cols = self.recording_ts.drop(self.classes, axis=1)

        # convert feature cols and recording ts to numpy array
        self.labels = EEGDataUtils.pd_to_np(self.recording_ts, cols_to_maintain=self.classes)
        self.feature_cols = EEGDataUtils.pd_to_np(self.feature_cols, cols_to_maintain='all')

    def __len__(self):
        """
        Total number of samples
        """
        return len(self.list_ids)

    def __getitem__(self, index):
        """
        Selects desired sample
        """

        ID = self.list_ids[index]
        print("ID ", ID)
        
        x = torch.tensor(self.feature_cols[ID], dtype=torch.float32)
        y = torch.tensor(self.labels[ID]).float()

        return x, y 
        


