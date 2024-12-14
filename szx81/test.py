from os import path as path
import core as co
co.set_test_data_dir('WikiText2')
co.TEST_DATA_DIR

train_txt = path.join(co.TEST_DATA_DIR, co.TEST_DATA, 'train.txt')
test_txt = path.join(co.TEST_DATA_DIR, co.TEST_DATA, 'test.txt')
data_store = path.join(co.TEST_DATA_DIR, co.TEST_DATA, co.DATA_STORE)

class TrainIter:
    def __init__(self, data_file):
        self.tokens = []
        self.data_file = data_file
        self.line_iterator = None
    
    def __iter__(self):
        self.file = open(self.data_file, 'r')
        self.line_iterator = iter(self.file.readline, '')
        return self

    def __next__(self): # Python 2: def next(self)
        if len(self.tokens) == 0:
            while len(self.tokens) == 0:
                line = next(self.line_iterator)
                if line == '':
                    raise StopIteration
                self.tokens = line.split()

        return self.tokens.pop(0)
    
train_iter = iter(TrainIter(train_txt))

for i in range(200):
    print(next(train_iter))