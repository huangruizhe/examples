import os
import torch
import pickle
import gzip


class Dictionary(object):
    def __init__(self, path):
        if path is None:
            self.word2idx = {}
            self.idx2word = []
        else:
            self.load_from_pkl(path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def load_from_pkl(self, path):
        with open(path, 'rb') as handle:
            self.word2idx, c2idx, max_idx = pickle.load(handle)
        self.word2idx['<eos>'] = max_idx
        max_idx += 1

        idx2c = dict()
        for c, id in c2idx.items():
            idx2c[id] = c

        self.idx2word = [None] * max_idx
        for w, id in self.word2idx.items():
            if id in idx2c:
                if self.idx2word[id] is None:
                    self.idx2word[id] = "word_freq_%d" % idx2c[id]
            else:
                self.idx2word[id] = w


class Corpus(object):
    def __init__(self, path, dict_path=None):
        self.dictionary = Dictionary(dict_path)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'dev.txt'))
        self.test = self.tokenize(os.path.join(path, 'dev.txt'))  # TODO

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # # Add words to the dictionary
        # with open(path, 'r', encoding="utf8") as f:
        #     tokens = 0
        #     for line in f:
        #         if len(line.strip()) == 0:
        #             continue
        #         words = line.strip().split() + ['<eos>']
        #         tokens += len(words)
        #         for word in words:
        #             self.dictionary.add_word(word)

        UNK = "<UNK>"
        unk_id = self.dictionary.word2idx[UNK]

        # count tokens:
        with gzip.open(path, mode='rt', encoding="utf8") if path.endswith('.gz') \
                else open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                line = line.strip()
                if line.startswith("###") or len(line) == 0:
                    continue
                words = line.split() + ['<eos>']
                tokens += len(words)

        # Tokenize file content
        with gzip.open(path, mode='rt', encoding="utf8") if path.endswith('.gz') \
                else open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = line.strip()
                if line.startswith("###") or len(line) == 0:
                    continue
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx.get(word, unk_id)
                    token += 1

        return ids
