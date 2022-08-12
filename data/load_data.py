import os
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader']
logger = logging.getLogger('MSA')
class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'sims3l':self.__init_sims,
        }
        DATA_MAP[args.datasetName]()

    def __init_sims(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        # Control Number of Supvised Data
        if self.args.supvised_nums != 2722:
            if self.mode == 'train':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    temp_data[self.mode][key] = data[self.mode][key][-self.args.supvised_nums:]
                data[self.mode] = temp_data[self.mode]
            
            if self.mode == 'valid':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    p = int(self.args.supvised_nums / 2)
                    temp_data[self.mode][key] = data[self.mode][key][-p:]
                data[self.mode] = temp_data[self.mode]
            
            if self.mode == 'train_mix':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    data_sup = data[self.mode][key][2722-self.args.supvised_nums:2722]
                    data_unsup = data[self.mode][key][2723:]
                    temp_data[self.mode][key] = np.concatenate((data_sup, data_unsup), axis = 0) 
                data[self.mode] = temp_data[self.mode]
        # Complete Data
        # if not self.mode == 'train_mix':
        #     self.rawText = data[self.mode]['raw_text']
        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.ids = data[self.mode]['id']
        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']
        # Labels
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims3l':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if self.mode == 'train_mix':
            self.mask = data[self.mode]['mask']

        # Clear dirty data
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0
        # Mean feture
        if  self.args.need_normalized:
            self.__normalize()

    def __normalize(self):
        self.vision_temp = []
        self.audio_temp = []
        for vi in range(len(self.vision_lengths)):
            self.vision_temp.append(np.mean(self.vision[vi][:self.vision_lengths[vi]], axis=0))
        for ai in range(len(self.audio_lengths)):
            self.audio_temp.append(np.mean(self.audio[ai][:self.audio_lengths[ai]], axis=0))
        self.vision = np.array(self.vision_temp)
        self.audio = np.array(self.audio_temp)

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def __getitem__(self, index):
        sample = {
            'index': index,
            # 'raw_text': self.rawText[index] if self.mode != 'train_mix' else [],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'audio_lengths': self.audio_lengths[index],
            'vision_lengths': self.vision_lengths[index],
            'mask' : self.mask[index] if self.mode == 'train_mix' else [],
        } 
        return sample

def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'train_mix': MMDataset(args, mode='train_mix'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test'),
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle= True)
        for ds in datasets.keys()
    }

    return dataLoader