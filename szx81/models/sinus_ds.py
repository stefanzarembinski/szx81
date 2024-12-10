import random
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
from nn_tools.data_sources import SinusDs
    
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
                 future_count=5,
                 context_len = 10,
                 hidden_size=100,
                 num_layers=3,
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
            seq_len=context_len,
            future_count=future_count,  
            end_index=5000, 
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
        # count = self.context_seq.seq_len + self.context_seq.future_count
        # data, _ = self.data_source.get_data(2 * count, count)
        # self.model.output_size = len(data[0])
    
    def get_training(self, data_count=1000, end_index=None, verbose=False):
        if end_index is not None:
            self.context_seq.end_index = end_index
        dt = self.context_seq.get_training(
                                data_count, end_index, verbose)
        return dt.features, dt.targets

    def preprocessing(self, x, y):
        x = Variable(torch.Tensor(x))
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        y = y.reshape(-1, 1)
        y = Variable(torch.Tensor(y))
        return x, y

    def train(self, data_count=1000, end_index=None, verbose=None):
        if verbose is None:
            verbose = self.verbose

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

            if verbose:
                if (epoch + 1) % 10 == 0:
                    print(
                f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.1e},  Diff: {1 - loss.item() / prev_loss:.1e}')
            
            prev_loss = loss.item()
    
    # def prediction(self, x, y):
    #     x, y = self.preprocessing(x, y)
    #     predicted = self.model(x).detach().numpy() # forward pass
    #     return predicted
         
    def show_action(self, shift=50, data_count=150):
        dt = self.context_seq.get_testing(
            context_count=self.model.num_layers * self.model.input_size,
            dist_count=shift, 
            data_count=data_count) 
        x = dt.features
        y = dt.targets
        
        x, y = self.preprocessing(x, y)
        predicted = self.model(x).detach().numpy() # forward pass
        self.context_seq.data_source.plot_prediction(
            dt = dt,
            predictions=(dt.indexes_tf, predicted), 
            data_count=20)

def test():
    sinus = lambda j, noise: .5 * math.sin(j * .03) + random.uniform(
        -noise, noise) + .5
    
    dr = NnDriver(
        data_source=SinusDs(
            sinus, 
            (None, None),
            step=3,
            noise=0.03
            ),
        model_class=Model,
        future_count=1,
        verbose=True
        )
    dr.train(data_count=500, end_index=6000)
    dr.show_action(shift=250)

def main():
    test()
    
if __name__ == "__main__":
    main()  


 