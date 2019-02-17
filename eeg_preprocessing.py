import numpy as np
import pandas as pd

class EEGPreprocessing():

    def __init__(self):
        pass
    
    @staticmethod
    def get_class_ranges(class_mapping, recording_types, recording_positions, recording_durations):

        class_ranges = {}

        for klass_name, klass_id in class_mapping.items():
            target_ids = np.argwhere(recording_types == klass_id)  
            durations = recording_durations[target_ids]
            starts = recording_positions[target_ids]
            ends = np.add(starts, durations)

            class_ranges[klass_name] = (starts, ends)

        return class_ranges
    
    @staticmethod
    def label_recording(recording_ts, class_ranges):

        new_col = pd.DataFrame(0, index=np.arange(recording_ts.shape[0]), columns=['class_label'])
        # add new column to the existing recording
        recording_ts = recording_ts.join(new_col)

        for klass_name, klass_range in class_ranges.items():

            starts, ends = klass_range
            for idx, _ in enumerate(starts):
                # Get all keys in the range

                keys_in_range = recording_ts[(recording_ts.index > int(starts[idx])) & (recording_ts.index < int(ends[idx]))].index.values
            
                recording_ts.loc[keys_in_range, 'class_label'] = klass_name

        return recording_ts