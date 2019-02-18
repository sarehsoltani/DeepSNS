from torch.utils.data import Dataset


class EEGData(Dataset):

    def __init__(self, list_ids, labels):
        self.list_ids = list_ids
        self.labels = labels

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
        # x = load(ID)
        x = 0
        y = self.labels[ID]

        return x, y
        
