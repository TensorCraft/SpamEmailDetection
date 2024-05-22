from BPE import BPE
import pickle

class Tokenizer():
    def __init__(self, OOV="<OOV>", PAD="<PAD>", maxLength=250, vocab_size=30000):
        self.OOV = OOV
        self.PAD = PAD
        self.maxLength = maxLength
        self.vocab_size = vocab_size
        self.vocab = {}
        self.index2token = {}

    def fit_texts_in_files(self, files):
        contents = []
        for file in files:
            contents.append(file.read())
        self.vocab, self.index2token = BPE(contents, vocab_size=self.vocab_size, OOV=self.OOV, PAD=self.PAD, maxLength=self.maxLength)

    def tokenize(self, text):
        tokens = text.split()[:self.maxLength]
        token_ids = [self.vocab.get(token, self.vocab[self.OOV]) for token in tokens]
        return token_ids

    def pad(self, text):
        token_ids = self.tokenize(text)
        if len(token_ids) >= self.maxLength:
            return token_ids[:self.maxLength]
        else:
            padded_token_ids = token_ids + [self.vocab[self.PAD]] * (self.maxLength - len(token_ids))
            return padded_token_ids

    def save_vocab(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)

    def load_vocab(self, filename):
        with open(filename, 'rb') as f:
            self.vocab = pickle.load(f)
            self.index2token = {index: token for token, index in self.vocab.items()}
