import torch
print(torch.__version__)
import torchtext
print(torchtext.__version__)
# ``torchtext`` is incompatible with ``torch 2.5.0``
# Now, I have installed ``torch 2.3.0`` which is incompatible 
# with existing ``torchaudio`` and ``torchvision``

# import torchaudio
# print(torchaudio.__version__)



import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("You can now install TorchText using pip!")
tokens
# ['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']



torchtext.disable_torchtext_deprecation_warning()

from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def build_and_save_vocabulary(train_iter, vocab_path='vocab.pt', min_freq=4):
    """
    Build a vocabulary from the training data iterator and save it to a file.
    
    Args:
        train_iter (iterator): An iterator over the training data.
        vocab_path (str, optional): The path to save the vocabulary file. Defaults to 'vocab.pt'.
        min_freq (int, optional): The minimum frequency of a word to be included in the vocabulary. Defaults to 4.
    
    Returns:
        torchtext.vocab.Vocab: The built vocabulary.
    """
    # Get the tokenizer
    tokenizer = get_tokenizer("basic_english")
    
    # Build the vocabulary
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'], min_freq=min_freq)
    
    # Set the default index to the unknown token
    vocab.set_default_index(vocab['<unk>'])
    
    # Save the vocabulary
    torch.save(vocab, vocab_path)
    
    return vocab



from os import path as path
import core as co
co.set_test_data_dir('WikiText2')
co.TEST_DATA_DIR

train_txt = path.join(co.TEST_DATA_DIR, co.TEST_DATA, 'train.txt')
test_txt = path.join(co.TEST_DATA_DIR, co.TEST_DATA, 'test.txt')
vocab_pt = path.join(
    co.TEST_DATA_DIR, co.TEST_DATA, co.DATA_STORE, 'vocab.pt')



class TrainIter:
    def __init__(self, data_file):
        self.tokens = []
        self.data_file = data_file
        self.line_iterator = None
        self.counter = None
    
    def __iter__(self):
        self.file = open(self.data_file, 'r', encoding = 'cp850')
        self.line_iterator = iter(self.file.readline, '')
        return self

    def __next__(self):
        if len(self.tokens) == 0:
            while len(self.tokens) == 0 or len(self.tokens) == 0:
                line = next(self.line_iterator)
                if line == '':
                    raise StopIteration
                self.tokens = line.split()
        token = self.tokens.pop(0)
        return token
        

# Assuming you have a training data iterator named `train_iter`
train_iter = iter(TrainIter(train_txt))

vocab = build_and_save_vocabulary(
    train_iter,
    vocab_path=vocab_pt
)

# You can now use the vocabulary
print(len(vocab))  # 23652
print(vocab(['ebi', 'AI'.lower(), 'qwerty']))  # [0, 1973, 0]
    