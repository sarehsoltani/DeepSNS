from torch.utils.data import Dataset
from mne import io
import numpy as np
import os
from sklearn.model_selection import train_test_split
from eeg import EEG
from eeg_preprocessing import EEGPreprocessing as EEGPrep
import os


class EEGDataUtils:

    def __init__(self):
        pass

    @staticmethod
    def prepare_partition(eeg_csv, val_split=0.7):
        
        partition = {}
        
        train, val = train_test_split(eeg_csv, test_size=val_split)

        partition['train'] = train.index.values
        partition['val'] = val.index.values

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

        return recording_ts_labeled


class EEGData(Dataset):

    def __init__(self, list_ids, labels):
        self.list_ids = list_ids
        self.labels = labels
        self.recording_ts_labeled = EEGDataUtils.prepare_eeg_csv()

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

        # load ID from the desired csv file
        x = self.recording_ts_labeled[ID]
        y = self.labels[ID]

        return x, y
        


