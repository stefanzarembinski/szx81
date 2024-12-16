import math
import numpy as np
np.set_printoptions(formatter={'float_kind':"{:-.3e}".format})
np.random.seed(0)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5, 4)

import torch
torch.manual_seed(0)

import torch.nn as nn
from torch.autograd import Variable

import core.hist_data as hd
from models.lstm.lstm_model import Model, NnDriver
from nn_tools.data_sources import CandleVsOpenDs

def test():
    data = list(hd.DICT_DATA.values())
    DataSource = CandleVsOpenDs

    verbose = False

    ds = DataSource(
            (MinMaxScaler(), MinMaxScaler(), MinMaxScaler(), MinMaxScaler()),
            step=3,
            verbose=verbose
            )

    dr = NnDriver(
        data_source=ds,
        model_object=Model(
            hidden_size=100, 
            num_layers=3,
            ),
        future_count=50,
        num_epochs=1000,
        accuracy=1e-5,
        learning_rate=0.001,
        verbose=verbose
        )
    
    if verbose > 0:
        print(dr)

    ds.fit_data(data[:20000])

    training_end_index = 10000
    training_data = data[:training_end_index]
    dr.train(
        data_count=500, 
        end_index=len(training_data) - dr.future_count,
        verbose=1 # verbose
        )
    
    shift = 50

    testing_end_index = training_end_index + shift
    shift += 10
    testing_data = data[:testing_end_index] 

    dr.context_seq.data_source.data = testing_data 
    dr.show_action(
    end_index=testing_end_index,
    data_count=150,
    future_data=dr.context_seq.data_source.future_data(
                            data, 
                            testing_end_index-1, 
                            length=50),
    show_features=True,
    verbose=verbose)

def main():  
    test()
    
if __name__ == "__main__":
    main()  


 