import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 4)

import torch
torch.manual_seed(0)

import torch.nn as nn
from torch.autograd import Variable

from core import _
from nn_tools.data_sequencer import ContextSequencer

class Model(nn.Module):
    def __init__(self, hidden_size, num_layers, 
                 input_size=None, output_size=None):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def init(self):
        self.model = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            batch_first=True)
        self.fc =  nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()        

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, (h_n, c_n) = self.model(x, (h_0.detach(), c_0.detach()))
        out = self.fc(out[:, -1, :])  # Selecting the last output

        return out

class NnDriver:
    def __init__(self,
                 data_source,
                 model_object,
                 future_count=5, # time horizon counted with forex period
                 seq_len=10, # number of data units feeding each LSTM module
                 num_epochs=1000,
                 accuracy=1e-5,
                 learning_rate=0.001,
                 verbose = False
        ):
        self.future_count = future_count
        self.num_epochs = num_epochs
        self.accuracy = accuracy
        self.learning_rate = learning_rate
        self.context_seq = ContextSequencer(
            data_source = data_source,
            seq_len=seq_len,  # model_object.input_size // data_source.feature_size,
            future_count=future_count,  
            end_index=5000
        )
        self.verbose = verbose

        self.data_source = self.context_seq.data_source
        self.data_source.verbose = self.verbose

        self.model = model_object
        self.model.input_size \
            = self.data_source.feature_size * self.context_seq.seq_len
        self.model.output_size = self.data_source.target_size 
        self.model.init()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.__setup()
        
    def __setup(self):
        pass
        # count = self.context_seq.seq_len + self.context_seq.future_count
        # data, _ = self.data_source.get_data(2 * count, count)
        # self.model.output_size = len(data[0])
    
    def get_training(self, data_count=1000, end_index=None, verbose=False):
        if end_index is not None:
            self.context_seq.end_index = end_index
        dt = self.context_seq.get_training(
                                data_count, end_index, verbose)
        return dt.features, dt.targets

    def __str__(self):
        st = f'''
input size: {self.model.input_size}
output size: {self.model.output_size}
LSTM:  {str(self.model)}
'''
        return st

    def preprocessing(self, x, y):
        x = Variable(torch.Tensor(x))
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        y = y.reshape(-1, 1)
        y = Variable(torch.Tensor(y))
        return x, y

    def train(self, data_count=1000, end_index=None, verbose=0):

        x, y = self.get_training(data_count, end_index, verbose=verbose)
        x, y = self.preprocessing(x, y)
        
        loss0 = None
        prev_loss = None
        for epoch in range(self.num_epochs):
            outputs = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, y)  
            loss.backward()
            if loss0 is None:
                loss0 = loss.item()

            if loss.item() / loss0 < self.accuracy:
                break
            if prev_loss is not None \
                and math.fabs(1 - loss.item() / prev_loss) < self.accuracy:
                break
            
            self.optimizer.step()

            if verbose > 1:
                if (epoch + 1) % 10 == 0:
                    print(
                f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.1e},  Diff: {1 - loss.item() / prev_loss:.1e}')
            
            prev_loss = loss.item()

        if verbose > 0:
            print(f'''
Loss decreased from {loss0:.1e} till {prev_loss:.1e}''')
         
    def show_action(self,
                    end_index, 
                    data_count=150,
                    future_data=30,
                    show_features=True,  
                    verbose=False):
        dt = self.context_seq.get_testing(
            end_index=end_index,
            data_count=data_count,
            context_count=self.model.num_layers * self.model.input_size,
            verbose=verbose)
        
        x = dt.features
        y = dt.targets
        
        x, y = self.preprocessing(x, y)
        predicted = self.model(x).detach().numpy() # forward pass
        self.context_seq.data_source.plot_prediction(
            dt = dt,
            predictions=(dt.indexes_tf, predicted),
            future_data=future_data, 
            show_features=show_features)
