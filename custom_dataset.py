import torch
from torch.utils.data import Dataset


class DANDataset(Dataset):
    def __init__(self, np_array, np_idx_len, label):
        super(DANDataset, self).__init__()
        self.np_array = np_array
        self.np_idx_len = np_idx_len
        self.label = label

    def __getitem__(self, index):
        return torch.LongTensor(self.np_array[index]), self.np_idx_len[index], torch.tensor(self.label[index],
                                                                                            dtype=torch.long)

    def __len__(self):
        return self.np_array.shape[0]


class DecPretrainDataset(Dataset):
    def __init__(self, np_array):
        super(DecPretrainDataset, self).__init__()
        self.np_array = np_array

    def __getitem__(self, index):
        return torch.FloatTensor(self.np_array[index])

    def __len__(self):
        return self.np_array.shape[0]


class DecDataset(Dataset):
    def __init__(self, np_array, np_labels):
        super(DecDataset, self).__init__()
        self.np_array = np_array
        self.np_labels = np_labels

    def __getitem__(self, index):
        if self.np_labels is None:
            return torch.FloatTensor(self.np_array[index])
        else:
            return torch.FloatTensor(self.np_array[index]), torch.tensor(self.np_labels[index])

    def __len__(self):
        return self.np_array.shape[0]


class GanDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.x_data = X
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y


class AnnDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.x_data = X
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        return x, y