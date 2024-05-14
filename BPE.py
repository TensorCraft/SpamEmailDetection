import json
import pandas as pd
import re

class BPE_Tokenizer:

    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.special_tokens = special_tokens if special_tokens else []
        
    def build_vocab_from_text(self, text):
        """Build initial vocabulary from text."""
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lower
        words = re.findall(r'\w+', text)
        vocab = {}
        for word in words:
            word = ' '.join(list(word)) + ' </w>'  # Add token end marker
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        for token in self.special_tokens:  # Add special tokens
            vocab[token + ' </w>'] = float('inf')  # Assign high frequency to ensure inclusion
        self.vocab = vocab

    def get_stats(self):
        """Calculate frequency of pairs of symbols in the vocabulary."""
        pairs = {}
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                if pair in pairs:
                    pairs[pair] += freq
                else:
                    pairs[pair] = freq
        return pairs

    def merge_vocab(self, pair):
        """Merge the most frequent pair in the vocabulary."""
        v_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in self.vocab:
            w_out = word.replace(bigram, replacement)
            v_out[w_out] = self.vocab[word]
        self.vocab = v_out

    def train_bpe(self):
        """Train BPE model by merging frequent pairs."""
        for i in range(self.vocab_size - len(self.vocab)):
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair)
        return self.vocab
    
        # """Train BPE model by merging frequent pairs until reaching the desired vocabulary size."""
        # j = 0
        # while len(self.vocab) > self.vocab_size:
        #     pairs = self.get_stats()
        #     if not pairs:
        #         break
        #     best_pair = max(pairs, key=pairs.get)
        #     self.merge_vocab(best_pair)
        #     j += 1
        #     print(j)
        # return self.vocab

    def tokenize(self, text):
        # """Tokenize new text using the BPE vocab."""
        # text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lower
        # words = re.findall(r'\w+', text)
        # tokens = []
        # for word in words:
        #     word = ' '.join(list(word)) + ' </w>'
        #     for subword in self.vocab:
        #         subword = subword.replace(' ', '')
        #         while subword in word:
        #             tokens.append(subword)
        #             word = word.replace(subword, '', 1)
        #     if word.strip():
        #         tokens.append(word.strip().replace(' ', ''))
        # return 
    
        """Tokenize new text using the BPE vocab."""
        text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lower
        words = re.findall(r'\w+', text)
        tokens = []
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            word_out = word
            for subword, _ in sorted(self.vocab.items(), key=lambda x: -len(x[0])):
                subword_pattern = subword.replace(' ', '')
                while subword_pattern in word_out:
                    tokens.append(subword_pattern)
                    start = word_out.find(subword_pattern)
                    word_out = word_out[:start] + word_out[start+len(subword_pattern):]
            if word_out.strip():
                tokens.extend(word_out.strip().split())
        return tokens

    def export_vocab_to_json(self, file_path):
        """Export vocabulary to a JSON file, each entry on a single line."""
        with open(file_path, 'w') as f:
            for key, value in self.vocab.items():
                f.write(json.dumps({key: value}) + '\n')  # Write each entry as a JSON object on a new line


    def import_vocab_from_json(self, file_path):
        # from a JSON file where each line is a separate JSON object."
        self.vocab = {}
        with open(file_path, 'r') as f:
            for line in f:
                self.vocab.update(json.loads(line))

def read_and_process_csv(file_path, text_column):
    df = pd.read_csv(file_path)
    print('length = ', len(df))
    train_size = int(len(df) * 1)
    train_df = df.sample( n= train_size, random_state = 42)
    train_df.reset_index(drop = True)
    # train_df.drop(train_df.columns[[0]], axis = 1, inplace = True)
    train_df = train_df[train_df.iloc[:, 0] == 1]
    train_df.to_csv('temp.csv', index = False)
    
    print('train_df exported!')

    # text_data = train_df[text_column].dropna().astype(str).tolist()
    text_data = ' '.join(train_df[text_column].dropna().tolist())  # Combine all text data
    # print(text_data)
    return text_data


# Specify file path and column containing the text
file_path = 'combined_data.csv'
text_column = 'text'

# Create the training dataset

# Read and process CSV
text_data = read_and_process_csv(file_path, text_column)


# Initialize BPE tokenizer
special_tokens = ['<s>', '</s>', '<unk>', '<pad>']
bpe_tokenizer = BPE_Tokenizer(vocab_size= 200, special_tokens=special_tokens)
bpe_tokenizer.build_vocab_from_text(text_data)
bpe_vocab = bpe_tokenizer.train_bpe()
print(bpe_vocab)

# Example tokenization
sample_text = "Hi my name is soo wee lim hahahahahahahaha."
tokens = bpe_tokenizer.tokenize(sample_text)
print("Tokens:", tokens)

# Export vocabulary to JSON
bpe_tokenizer.export_vocab_to_json('vocab.json')

# Import vocabulary from JSON
bpe_tokenizer.import_vocab_from_json('vocab.json')
print("Vocabulary Imported and Ready to Use")