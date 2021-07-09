import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class TranscriptLoader(pl.LightningDataModule):
    def __init__(self, folder, max_seq_len, num_workers, max_transcripts_per_batch, collate_fn):
        super().__init__()
        
        self.train = None
        self.val = None
        self.test = None
     
        self.folder = folder
        self.num_workers = num_workers
        self.max_transcripts_per_batch = max_transcripts_per_batch
        self.collate_fn = collate_fn
        self.max_seq_len = max_seq_len

    def setup(self, val_set=None, test_set=None):
        # merge all data sets together
        files = os.listdir(self.folder)
        
        if val_set:
            assert val_set in files, f'{val_set} not in {self.folder}'
            files.remove(val_set)
            print(f'Validation set: {val_set}')
            data = np.load(os.path.join(self.folder, val_set), allow_pickle=True)
            self.val = self.load_data(data)
        
        if test_set:
            assert test_set in files, f'{test_set} not in {self.folder}'
            files.remove(test_set)
            print(f'Test set: {test_set}')
            data = np.load(os.path.join(self.folder, test_set), allow_pickle=True)
            self.test = self.load_data(data)
        
        if len(files) > 0:
            data = [np.load(os.path.join(self.folder, file), allow_pickle=True) for file in files]
            data = np.vstack(data)
            self.train = self.load_data(data)

    def load_data(self, data):
        # reorganize data into tuples: (x, y, transcript name)
        data_samples = zip([d[:,0] for d in data[:,0]], [d[:,1] for d in data[:,0]], data[:,1])
        data_samples = np.array(sorted(data_samples, key=lambda x: x[0].shape[0]), dtype=object)
        
        return data_samples
    
    def shuffle(self, data=None):
        # local shuffle of sample with respects to transcript lens (mimicks bucket sampling)
        data = np.vstack(data)
        lens = np.array([ts[0].shape[0] for ts in data])
        splits = np.arange(1,max(lens),400)
        idxs = np.arange(lens.shape[0])
        shuffled_idxs = []
        for l, u in zip(splits, np.hstack((splits[1:],[99999]))):
            mask = np.logical_and(l < lens, lens < u)
            shuffled = idxs[mask]
            np.random.shuffle(shuffled)
            shuffled_idxs.append(shuffled)
        shuffled_idxs = np.hstack(shuffled_idxs)

        data = data[shuffled_idxs]
        lens = np.array([ts[0].shape[0] for ts in data])
        
        # split idx sites l
        l = []
        # idx pos
        idx_pos = 0
        
        while len(data) > idx_pos:
            # get lens of leftover transcripts
            lens_set = lens[idx_pos:]
            # calculate memory based on number of samples and max length (+2 for transcript start/stop token)
            mask = (lens_set+2)*(np.arange(lens_set.shape[0])+1) >= (self.max_seq_len)
            mask_idx = np.where(mask)[0]
            # get idx to split
            if len(mask_idx) > 0 and (mask_idx[0] > 0):
                # max amount of transcripts per batch
                idx = min(mask_idx[0],self.max_transcripts_per_batch)
                idx_pos += idx
                l.append(idx_pos)       
            else:
                break
        return np.split(data, l)[:-1]

    def train_dataloader(self):
        return DataLoader(self.shuffle(self.train), batch_size=1, collate_fn=self.collate_fn, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.shuffle(self.val), batch_size=1, collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.shuffle(self.test), batch_size=1, collate_fn=self.collate_fn, num_workers=self.num_workers)
    
def collate_fn(batch):
    lens = np.array([len(s[0]) for s in batch[0]])
    max_len = max(lens)
    tr_labels = batch[0][:,2]
    
    # Add padding
    x_b = torch.LongTensor(np.array([np.concatenate(([5], x, [6], [4]*l)) for x, l in zip(batch[0][:,0], max_len - lens)], dtype=np.int))
    y_b = torch.LongTensor(np.array([np.concatenate(([2], y, [2], [4]*l)) for y, l in zip(batch[0][:,1], max_len - lens)], dtype=np.int))
    nt_mask = torch.LongTensor(np.array([np.concatenate(([False], np.full(x.shape, True), [False]*(l+1))) for x, l in zip(batch[0][:,1], max_len - lens)], dtype=np.int)).bool()
    
    # Create mask
    mask = np.zeros(x_b.shape, dtype=np.int)
    for i,l in enumerate(lens):
        mask[i,:l+2] = 1
    return x_b, y_b, tr_labels, torch.LongTensor(mask).bool(), nt_mask, lens
