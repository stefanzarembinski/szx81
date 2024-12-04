import sys
import os
from os import path
import importlib

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


class Preprocessor:
    def __init__(self):
        self.ss = StandardScaler()
        self.mm = MinMaxScaler()

    def pre(self, x, y=None):
        x = self.ss.fit_transform(x)
        x = Variable(torch.Tensor(x))
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))

        if y is None:
            return x
        
        y = y.reshape(-1, 1)
        y = self.mm.fit_transform(y)
        y = Variable(torch.Tensor(y))
        return x, y
    
    def inverse_x(self, predicted):
        return self.mm.inverse_transform(predicted)

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
                 data_source_class,
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

        
        self.preprocessor = Preprocessor()
        self.num_epochs = num_epochs
        self.accuracy = accuracy
        self.learning_rate = learning_rate
        self.context_seq = ds.ContextSequencer(
            data_source_class = data_source_class, 
            end_day=2, 
            seq_len=context_len, 
            future_len=future_len
        )  
        self.verbose = verbose
        self.model = model_class(
            input_size=context_len, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=1
            )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate) 
    
    def get_training(self, end_day=None, data_count=1000, verbose=False):
        if end_day is not None:
            self.context_seq.end_index = end_day * 60 * 24
        x, y = self.context_seq.get_training(
                                data_count, verbose)
        return x, y

    def train(self, end_day=None, data_count=1000):
        x, y = self.get_training(end_day, data_count, verbose=self.verbose)
        x, y = self.preprocessor.pre(x, y)
        
        loss0 = None
        for epoch in range(self.num_epochs):
            outputs = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, y)  
            loss.backward()
            if loss0 is None:
                loss0 = loss.item()
            if loss.item() / loss0 < self.accuracy:
                break
            self.optimizer.step()

            if self.verbose:
                if (epoch + 1) % 10 == 0:
                    print(
                f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.1e}')
    
    def prediction(self, x, y):
        predicted = self.model(
        self.preprocessor.pre(x)).detach().numpy()#forward pass

        self.preprocessor.pre(x, y)
        predicted = self.preprocessor.inverse_x(predicted)
        return predicted[-1][-1]
    
    def show_action(self, shift=50):
        end_index = self.context_seq.last_trained_index + shift
        fact = []
        pred = []
        for i in range(20):
            future_index = end_index + self.context_seq.future_len + i
            current_index = end_index + i
            data, indexes = self.context_seq.data_source.get_data(end_index=future_index, count=0)
            fact.append(data[0])
            x, y, _ = self.context_seq.create_sequences(
                current_index, self.context_seq.seq_len, 
                self.model.num_layers * self.model.input_size)
            pred.append(self.prediction(x, y))

        plt.plot(fact, label='fact', color='green')
        plt.plot(pred, label='prediction', color='blue')
        plt.legend()
        plt.show()

def test():
    hd.set_hist_data(data_count=None)

    dr = NnDriver(
        data_source_class=ds.ForexDataSource,
        model_class=Model,
        verbose=True
        )
    dr.train()
    dr.show_action(shift=250)

def main():
    test()
    
if __name__ == "__main__":
    main()  


