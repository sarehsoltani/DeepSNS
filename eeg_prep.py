from mne import io
import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

def get_class_positions(classes_dict, recording_types):

    class_positions = {}

    for klass_name, klass_id in classes_dict.items():
        class_positions[klass_name] = np.argwhere(recording_types == klass_id)

    return class_positions

def clear_label_columns(column):

    column.replace(0, 'none', inplace=True)

    print("Done replacing ")
    return column

root = './data'
classes = ['img_lhand', 'img_rhand', 'img_bfeet', 'img_tongue', 'none']

classes_dict = {
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

classes_ids = get_class_positions(classes_dict, recording_types)

# Add label column to recording ts dataframe
new_col = pd.DataFrame(0, index=np.arange(recording_ts.shape[0]), columns=['class_label'])

for klass, klass_ids in classes_ids.items():
    new_col.iloc[klass_ids] = klass

new_col = clear_label_columns(new_col)

print("New labels column: ", new_col)
# 
# recording_ts['class_label'] = pd
# for klass, klass_ids in classes_ids.items():
# 
    # print("Class {0} IDs: {1}".format(klass, klass_ids))

# print("Recording time-series: ", recording_ts)
# recording_idx = classes_ids['img_rhand'][0]
# start_time = recording_positions[recording_idx]
# end_time = start_time + recording_durations[recording_idx]
# print("One sample of class 1 is from {} to {}".format(start_time, end_time))
