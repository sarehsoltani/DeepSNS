from mne import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

def get_class_ranges(class_mapping, recording_types, recording_positions, recording_durations):

    class_ranges = {}

    for klass_name, klass_id in class_mapping.items():
        target_ids = np.argwhere(recording_types == klass_id)  
        durations = recording_durations[target_ids]
        starts = recording_positions[target_ids]
        ends = np.add(starts, durations)

        class_ranges[klass_name] = (starts, ends)

    return class_ranges

def clear_label_columns(column):

    column.replace(0, 'none', inplace=True)

    print("Done replacing ")
    return column

root = './data'
classes = ['img_lhand', 'img_rhand', 'img_bfeet', 'img_tongue', 'none']

class_mapping = {
    'img_lhand': 769, 
    'img_rhand': 770,
    'img_bfeet': 771,
    'img_tongue': 772,
}


bci_record_names = []

for file_name in os.listdir(root):
    bci_record_names.append(file_name)

# print("File names: ", bci_record_names)

# load one file from dir
gdf_file = io.read_raw_edf(root + '/' + bci_record_names[3], stim_channel=1, preload=True)

recording_headers = io.find_edf_events(gdf_file)

recording_positions = recording_headers[1]
recording_types = recording_headers[2]
recording_durations = recording_headers[4]

# convert to dataframe
recording_ts = gdf_file.to_data_frame()         # Recording time-series

# print(recording_positions)
class_ranges = get_class_ranges(class_mapping, recording_types, recording_positions, recording_durations)

new_col = pd.DataFrame(0, index=np.arange(recording_ts.shape[0]), columns=['class_label'])

# add new column to the existing recording
recording_ts = recording_ts.join(new_col)

for klass_name, klass_range in class_ranges.items():

    starts, ends = klass_range
    for idx, start in enumerate(starts):
        # Get all keys in the range

        keys_in_range = recording_ts[(recording_ts.index > int(starts[idx])) & (recording_ts.index < int(ends[idx]))].index.values
    
        recording_ts.loc[keys_in_range, 'class_label'] = klass_name


# Fill all nans with 'none'

recording_ts['class_label'].fillna('img_none', inplace=True)
recording_ts['class_label'].replace(0.0, 'img_none', inplace=True)

# recording_ts.to_csv('eeg_preped.csv')
# print(recording_ts.describe())

eeg0_ch = recording_ts[['EOG-right', 'EOG-central', 'EOG-left']]
eeg0_ch = eeg0_ch.astype(float)
eeg0_ch.plot()
plt.show()
# plt.show()
# plot channel1
# print(recording_ts.columns)