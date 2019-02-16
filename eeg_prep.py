from mne import io
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

def get_class_positions(classes_dict, recording_types):

    class_positions = {}

    for klass_name, klass_id in classes_dict.items():
        class_positions[klass_name] = np.argwhere(recording_types == klass_id)

    return class_positions

root = './data'
classes = ['img_lhand', 'img_rhand', 'img_bfeet', 'img_tongue']

classes_dict = {
    'img_lhand': 769, 
    'img_rhand': 770,
    'img_bfeet': 771,
    'img_tongue': 772
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
recording_pd = gdf_file.to_data_frame()

classes_positions = get_class_positions(classes_dict, recording_types)


recording_idx = classes_positions['img_rhand'][0]
start_time = recording_positions[recording_idx]
end_time = start_time + recording_durations[recording_idx]
print("One sample of class 1 is from {} to {}".format(start_time, end_time))