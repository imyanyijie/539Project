import h5py
import numpy as np
import torch
import torch.utils.data as data
import os
import json

class H5DirDataset(data.Dataset):

    def __init__(self, root_dir, pad_maxlen, cnn, data_start=0.0, data_end=1.0, use_perc=1.0):
        super(H5DirDataset, self).__init__()
        self.label_len = {}
        self.data_size = 0
        self.data_start = data_start
        self.data_end = data_end
        self.indices = []
        self.label_to_num = {}
        self.num_to_label = {}
        self.cnn = cnn

        self.pad_maxlen = pad_maxlen

        file_list = os.listdir(root_dir)
        for line in file_list:
            filepath = os.path.join(root_dir, line)
            if os.path.isdir(filepath):
                # read len
                with open(os.path.join(filepath, 'info.json'), 'r') as f:
                    len_dict = json.load(f)
                this_len = len_dict['len']
                # log label
                label = line.split('/')[-1]
                label_n = len(self.label_to_num)
                self.label_to_num[label] = label_n
                self.num_to_label[label_n] = label

                # calculate indices
                start_point = int(this_len * data_start)
                end_point = int(this_len * data_end)
                picked_indices = np.random.choice(list(range(start_point, end_point)),
                                                  size=int((end_point - start_point) * use_perc),
                                                  replace=False).tolist()
                self.label_len[filepath] = len(picked_indices)
                self.data_size += len(picked_indices)
                self.indices.extend([(filepath, picked_index, label_n) for picked_index in picked_indices])
        np.random.shuffle(self.indices)

    def __getitem__(self, index):

        dir_name, index, label = self.indices[index]

        with h5py.File(os.path.join(dir_name, '%d.h5' % index), 'r') as f:
            data = np.array(f.get('data'))
        data = data.reshape(-1, 1, 256, 256)

        if self.cnn:
            data = data.reshape(-1, 1, 256, 256)[-1, :, :, :]

        # padding
        # data = self.__padding(data)

        return (torch.from_numpy(data).float(),
                torch.tensor(label).long())

    def __len__(self):
        return self.data_size

    def __padding(self, data):
        rnn_l = data.shape[0]
        if rnn_l >= self.pad_maxlen:
            return data[-self.pad_maxlen:, :, :, :]
        else:
            pad_zeros = np.zeros((self.pad_maxlen - rnn_l, 1, 256, 256))
            return np.concatenate([pad_zeros, data], axis=0)