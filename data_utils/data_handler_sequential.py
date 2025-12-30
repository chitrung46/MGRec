import pickle
import numpy as np
import scipy.sparse as sp
from config.configurator import configs
from data_utils.datasets_sequential import SequentialDataset, ReviewSequentialDataset
import torch
import torch.utils.data as data
from os import path


class DataHandlerSequential:
    def __init__(self):
        self.flag_review = False

        if configs['data']['name'] == 'ml-20m':
            predir = './datasets/sequential/ml-20m_seq/'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 'sports':
            predir = './datasets/sequential/sports_seq/'
            configs['data']['dir'] = predir
        elif configs['data']['name'] == 'music':
            predir = './datasets/sequential/music_seq/'
            configs['data']['dir'] = predir
            self.flag_review = True
            
        self.trn_file = path.join(predir, 'train.tsv')
        self.val_file = path.join(predir, 'test.tsv')
        self.tst_file = path.join(predir, 'test.tsv')
        self.max_item_id = 0

        if configs['data']['name'] in ['music']:
            self.trn_review_file = path.join(predir, 'train_reviews.tsv')
            self.tst_review_file = path.join(predir, 'test_reviews.tsv')

    def _read_tsv_to_user_seqs(self, tsv_file):
        user_seqs = {"uid": [], "item_seq": [], "item_id": []}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            # skip header
            line = f.readline()
            while line:
                uid, seq, last_item = line.strip().split('\t')
                seq = seq.split(' ')
                seq = [int(item) for item in seq]
                user_seqs["uid"].append(int(uid))
                user_seqs["item_seq"].append(seq)
                user_seqs["item_id"].append(int(last_item))

                self.max_item_id = max(
                    self.max_item_id, max(max(seq), int(last_item)))
                line = f.readline()
        return user_seqs
        
    def _read_tsv_to_review_seqs(self, tsv_file):
        review_seqs = {'uid': [], 'review_seq': []}
        with open(tsv_file, 'r') as f:
            line = f.readline()
            line = f.readline()
            i = 1
            while line:
                # print(i)
                # i += 1
                uid, seq, _ = line.strip().split('\t')
                seq = seq.split('||')
                review_seqs['uid'].append(int(uid))
                review_seqs['review_seq'].append(seq)

                line = f.readline()
        return review_seqs
                       
    def _set_statistics(self, user_seqs_train, user_seqs_test):
        user_num = max(max(user_seqs_train["uid"]), max(
            user_seqs_test["uid"])) + 1
        configs['data']['user_num'] = user_num
        # item originally starts with 1
        configs['data']['item_num'] = self.max_item_id

    def _seq_aug(self, user_seqs, review_seqs=None):
        if self.flag_review:
            user_seqs_aug = {"uid": [], "item_seq": [], "item_id": [], "review_seq": []}
            for uid, seq, last_item, review_seq in zip(user_seqs["uid"], user_seqs["item_seq"], user_seqs["item_id"], review_seqs["review_seq"]):
                user_seqs_aug["uid"].append(uid)
                user_seqs_aug["item_seq"].append(seq)
                user_seqs_aug["item_id"].append(last_item)
                user_seqs_aug["review_seq"].append(review_seq)
                for i in range(1, len(seq)-1):
                    user_seqs_aug["uid"].append(uid)
                    user_seqs_aug["item_seq"].append(seq[:i])
                    user_seqs_aug["item_id"].append(seq[i])
                    user_seqs_aug["review_seq"].append(review_seq[:i])
        else:
            user_seqs_aug = {"uid": [], "item_seq": [], "item_id": []}
            for uid, seq, last_item in zip(user_seqs["uid"], user_seqs["item_seq"], user_seqs["item_id"]):
                user_seqs_aug["uid"].append(uid)
                user_seqs_aug["item_seq"].append(seq)
                user_seqs_aug["item_id"].append(last_item)
                for i in range(1, len(seq)-1):
                    user_seqs_aug["uid"].append(uid)
                    user_seqs_aug["item_seq"].append(seq[:i])
                    user_seqs_aug["item_id"].append(seq[i])
        return user_seqs_aug

    @staticmethod
    def collate_fn(batch):
        zipped_batch = list(zip(*batch))
        uid = torch.tensor(zipped_batch[0], dtype=torch.long)
        item_seq = torch.stack(zipped_batch[1])
        last_item = torch.tensor(zipped_batch[2], dtype=torch.long)
        
        if len(zipped_batch) == 5:
            neg_item = torch.stack(zipped_batch[3])
            review_seq = list(zipped_batch[4])
            
            return uid, item_seq, last_item, neg_item, review_seq       
        else:
            review_seq = list(zipped_batch[3])
            
            return uid, item_seq, last_item, review_seq
    
    def load_data(self):
        user_seqs_train = self._read_tsv_to_user_seqs(self.trn_file)
        user_seqs_test = self._read_tsv_to_user_seqs(self.tst_file)
        self._set_statistics(user_seqs_train, user_seqs_test)

        review_seqs_train = None
        review_seqs_test = None

        if configs['data']['name'] in ['music']:
            review_seqs_train = self._read_tsv_to_review_seqs(self.trn_review_file)
            review_seqs_test = self._read_tsv_to_review_seqs(self.tst_review_file)

        # seqeuntial augmentation: [1, 2, 3,] -> [1,2], [3]
        if 'seq_aug' in configs['data'] and configs['data']['seq_aug']:
            user_seqs_aug = self._seq_aug(user_seqs_train, review_seqs_train)
            trn_data = ReviewSequentialDataset(user_seqs_train, user_seqs_aug=user_seqs_aug, flag_review=True)
        else:
            trn_data = SequentialDataset(user_seqs_train)
        tst_data = ReviewSequentialDataset(user_seqs_test, mode='test', flag_review=True, review_seqs=review_seqs_test)
        self.test_dataloader = data.DataLoader(
            tst_data, batch_size=configs['test']['batch_size'], shuffle=False, num_workers=8, collate_fn=self.collate_fn)
        self.train_dataloader = data.DataLoader(
            trn_data, batch_size=configs['train']['batch_size'], shuffle=True, num_workers=8, collate_fn=self.collate_fn)
