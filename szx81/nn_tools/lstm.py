import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 4)

import torch
torch.manual_seed(0)

import torch.nn as nn
from torch.autograd import Variable

import core as co
from core import _
import hist_data as hd
import nn_tools.data_sequencer as ds
import nn_tools.data_source as ns
    

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.model = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True)
        self.fc =  nn.Linear(hidden_size, output_size)
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
                 model_class,
                 future_len=5,
                 context_len = 10,
                 hidden_size=100,
                 num_layers=3,
                 num_epochs=1000,
                 accuracy=1e-5,
                 learning_rate=0.001,
                 verbose = False
        ):
        self.num_epochs = num_epochs
        self.accuracy = accuracy
        self.learning_rate = learning_rate

        self.context_seq = ds.ContextSequencer(
            data_source_class = data_source, 
            end_day=2, 
            seq_len=context_len, 
            future_len=future_len
        ) 
        self.model = model_class(
            input_size=context_len, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=1
            )
        self.data_source = self.context_seq.data_source
        
        self.verbose = verbose        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.__setup()
        
    def __setup(self):
        pass
        # count = self.context_seq.seq_len + self.context_seq.future_len
        # data, _ = self.data_source.get_data(2 * count, count)
        # self.model.output_size = len(data[0])
    
    def get_training(self, end_day=None, data_count=1000, verbose=False):
        if end_day is not None:
            self.context_seq.end_index = end_day * 60 * 24
        x, y = self.context_seq.get_training(
                                data_count, verbose)
        return x, y

    def train(self, end_day=None, data_count=1000, verbose=None):
        if verbose is None:
            verbose = self.verbose

        x, y = self.get_training(end_day, data_count, verbose=verbose)
        x = Variable(torch.Tensor(x))
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        y = y.reshape(-1, 1)
        y = Variable(torch.Tensor(y))
        
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

            if verbose:
                if (epoch + 1) % 10 == 0:
                    print(
                f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.1e},  Diff: {1 - loss.item() / prev_loss:.1e}')
            
            prev_loss = loss.item()
    
    def prediction(self, x, y):
        x, y = self.preprocessor.pre(x, y)
        predicted = self.model(x).detach().numpy() # forward pass
        predicted = self.context_seq.data_source.inverse_transform([predicted])
        return predicted

    def __prediction(self, shift=50, count=50):
        warmup_len = self.model.num_layers * self.model.input_size
        x, y, indexes_x, indexes_y  = self.context_seq.create_sequences(
            end_index=self.context_seq.last_trained_index + shift + count, 
            seq_len=self.context_seq.seq_len, 
            count=warmup_len + count
            )
        y_predicted = self.prediction(x, y)
        x_act = [_[-1] for _ in x]
        y_act = [_ for _ in y]
        y_pred = np.array([_[0] for _ in y_predicted])
        y_pred = y_pred \
            + (y_act[-self.context_seq.future_len - 1] - y_pred[-self.context_seq.future_len - 1])

        x_act = [[_[-1] for _ in indexes_x][warmup_len:], 
                                            x_act[warmup_len:]]
        y_act = [[_ for _ in indexes_y][warmup_len:],
                                            y_act[warmup_len:]]
        y_pred = [[_ for _ in indexes_y][warmup_len:], 
                                            y_pred[warmup_len:]]
        return x_act, y_act, y_pred
         
    def show_action(self, shift=50, count=50):
        x_act, y_act, y_pred = self.__prediction(shift, count)
        plt.plot(*y_pred, label='prediction', color='green', 
                 linewidth=7, alpha=0.5)     
        plt.plot(*x_act, label='feature', color='red', 
                 linewidth=5, alpha=0.5)
        plt.plot(*y_act, label='actual target', color='blue', 
                 linewidth=3, alpha=0.5)           
        plt.legend()
        plt.show()

def test():
    hd.set_hist_data(data_count=None)
    dr = NnDriver(
        data_source_class=ns.SinusDataSource,
        model_class=Model,
        verbose=True
        )
    dr.train()
    dr.show_action(shift=250)

def main():
    test()
    
if __name__ == "__main__":
    main()  


