from mne import io
import os

import warnings
warnings.filterwarnings("ignore")

root = './BCICIV_2a_gdf'
classes = ['img_lhand', 'img_rhand', 'img_bfeet', 'img_tongue']


bci_record_names = []

for file_name in os.listdir(root):
    bci_record_names.append(file_name)

# print("File names: ", bci_record_names)

# load one file from dir
gdf_file = io.read_raw_edf(root + '/' + bci_record_names[3], stim_channel=1, preload=True)

recording_headers = io.find_edf_events(gdf_file)

recording_positions = recordiزیادng_headers[1]
recording_types = recording_headers[2]
recording_durations = recording_headers[4]

# convert to dataframe
recording_pd = gdf_file.to_data_frame()

print("Recording positions: ", recording_positions)
print("Recording durations: ", recording_durations)
print("Recording types: ", recording_types)

# # show the raw headers
# print("Raw headers: ", recording_headers)

# # show the columns
# print("Columns: ", recording_pd.columns)

# # show the recording
# print(recording_pd)