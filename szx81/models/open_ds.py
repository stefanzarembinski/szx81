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

import hist_data as hd
from nn_tools.data_sources import OpenDs
from models.lstm_model import Model, NnDriver

def test():
    verbose = True
    training_end_index = 4000
    data = list(hd.DICT_DATA.values())
    training_data = data[:training_end_index]

    ds = OpenDs(
            training_data, 
            (MinMaxScaler(), MinMaxScaler()),
            step=3,
            verbose=verbose
            )

    # context_count = num_layers * input_size
    dr = NnDriver(
        data_source=ds,
        model_object=Model(
            input_size=10, 
            hidden_size=100, 
            num_layers=3, 
            output_size=1
            ),
        future_count=15,
        num_epochs=1000,
        accuracy=1e-5,
        learning_rate=0.001,
        verbose=verbose
        )
    
    dr.train(data_count=500, 
             end_index=len(training_data) - dr.future_count,
             verbose=verbose)

    shift = 50
    testing_end_index = training_end_index + shift
    testing_data = data[:testing_end_index] 
    
    dr.context_seq.data_source.data = testing_data
    dr.show_action(
    end_index=testing_end_index,
    data_count=50,
    future_data=dr.context_seq.data_source.future_data(
                            data, 
                            testing_end_index, 
                            length=30),
    show_features=True,
    verbose=verbose)

def main():  
    test()
    
if __name__ == "__main__":
    main()    


 