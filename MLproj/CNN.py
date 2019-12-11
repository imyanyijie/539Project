from tqdm import tqdm
import numpy as np
import os
import json
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import time
import random
from torch.autograd import Variable
import h5py
from collections import defaultdict

from H5Dataset import H5DirDataset
from eval_util import *

import apex
from apex import amp

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

np.random.seed(2019)


class CNM(nn.Module):
    def __init__(self, n_classes, seq_len=13):
        super(CNM, self).__init__()

        self.seq_len = seq_len

        # pretrained
        self.pretrained = models.resnet18(pretrained=True, progress=True)
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.pretrained.fc = nn.Linear(512, 256)

        # custom conv layer
        self.conv = nn.Conv2d(1, 3, (7, 7), stride=1, padding=2)

        self.fc_final = nn.Linear(256, n_classes)

    def forward(self, x):
        cnn_input = x
        pret_input = self.conv(cnn_input)
        cnn_out = F.relu(self.pretrained(pret_input))

        out = self.fc_final(cnn_out)

        return out


def train_model(dataloders, model, criterion, optimizer, num_epochs=25):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict()
    }
    torch.save(checkpoint, 'best_amp_checkpoint.pt')
    best_acc = 0.0
    dataset_sizes = {'train': len(dataloders['train'].dataset),
                     'valid': len(dataloders['valid'].dataset)}

    for epoch in range(num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloders[phase]):
                if use_gpu:
                    inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # return outputs, labels, preds
                loss = criterion(outputs, labels)

                if phase == 'train':
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data).cpu().item()
                # print(running_corrects)

            if phase == 'train':
                train_epoch_loss = running_loss / dataset_sizes[phase]
                train_epoch_acc = running_corrects / dataset_sizes[phase]
            else:
                valid_epoch_loss = running_loss / dataset_sizes[phase]
                valid_epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'valid' and valid_epoch_acc > best_acc:
                best_acc = valid_epoch_acc
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                }
                torch.save(checkpoint, 'best_amp_checkpoint.pt')

        print('Epoch [{}/{}] train loss: {:.8f} acc: {:.4f} '
              'valid loss: {:.8f} acc: {:.4f}'.format(
            epoch, num_epochs - 1,
            train_epoch_loss, train_epoch_acc,
            valid_epoch_loss, valid_epoch_acc))

    print('Best val Acc: {:4f}'.format(best_acc))

    checkpoint = torch.load('best_amp_checkpoint.pt')

    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])

    del inputs, labels

    return model, optimizer, amp


def test_model(dataloders, model, criterion):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    dataset_sizes = {'test': len(dataloders['test'].dataset)}

    for epoch in range(1):
        for phase in ['test']:

            model.train(False)

            running_loss = 0.0
            running_corrects = 0

            label_cpu = np.zeros(dataset_sizes['test'], )
            pred_cpu = np.zeros((dataset_sizes['test'], 5))
            n = 0
            for inputs, labels in tqdm(dataloders[phase]):
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                new_n = n + labels.size()[0]
                label_cpu[n:new_n] = labels.cpu()

                outputs = model(inputs)
                _, preds = torch.topk(outputs.data, 5)

                pred_cpu[n:new_n, :] = preds.cpu()

                loss = criterion(outputs, labels)

                n = new_n

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds[:, 0] == labels).cpu().item()

    del inputs, labels

    return model, label_cpu, pred_cpu


if __name__ == '__main__':
    N_CLASSES = 345

    BATCH_SIZE = 256
    LOADER_WORKERS = 4
    PERC = 95

    loader_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': LOADER_WORKERS}

    dir_path = "/mnt/data0/data/ndjson_p"
    lens_dist = np.load(os.path.join(dir_path, 'lens.npy'))

    SEQ_LEN = int(np.percentile(lens_dist, PERC))
    print('sequence length is %d' % SEQ_LEN)

    train_ds = H5DirDataset(dir_path, cnn=True, pad_maxlen=SEQ_LEN, data_start=0.0, data_end=0.95, use_perc=0.1)
    val_ds = H5DirDataset(dir_path, cnn=True, pad_maxlen=SEQ_LEN, data_start=0.95, data_end=0.975, use_perc=0.1)
    test_ds = H5DirDataset(dir_path, cnn=True, pad_maxlen=SEQ_LEN, data_start=0.975, data_end=1.0, use_perc=0.1)
    train_dl = data.DataLoader(train_ds, **loader_params)
    val_dl = data.DataLoader(val_ds, **loader_params)
    test_dl = data.DataLoader(test_ds, **loader_params)
    dataloders = {'train': train_dl,
                  'valid': val_dl,
                  'test': test_dl}

    model = CNM(n_classes=N_CLASSES, seq_len=SEQ_LEN)
    use_gpu = torch.cuda.is_available()
    print('gpu available:' + str(use_gpu))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    if use_gpu:
        model = model.cuda()

    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    checkpoint = torch.load('best_amp_checkpoint CNN.pt')

    #model.load_state_dict(checkpoint['model'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #amp.load_state_dict(checkpoint['amp'])

    start_time = time.time()
    best_model, best_optimizer, best_amp = train_model(dataloders, model, criterion, optimizer, num_epochs=20)
    print('Training time: {:10f} minutes'.format((time.time() - start_time) / 60))

    model.eval()
    model, label_cpu, pred_cpu = test_model(dataloders, model, criterion)


    cm = confusion_matrix(label_cpu, pred_cpu[:, 0])
    plt.figure(figsize=(18, 18))
    plt.imshow(cm, cmap='hot')
    plt.title('Confusion Matrix')
    plt.show()


    with open('/mnt/data0/data/dir_num_to_name.json', 'r') as f:
        dir_num_to_name = json.load(f)
    model.eval()
    for i in range(1, 6):
        print('accuracy of top %d confident prediction: %.4f' % (i, acc_topk(pred_cpu, label_cpu, i)))

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict()
    }
    torch.save(checkpoint, 'best_amp_checkpoint.pt')
