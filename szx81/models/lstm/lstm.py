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
from models.lstm.lstm_model import Model, NnDriver
from nn_tools.data_sources import SinusDs

def test():
    training_end_index = 4000
    data = SinusDs.Sinus(noise=0.03, stop=training_end_index)
    

    ds = SinusDs(
            data, 
            (None, None),
            step=3,
            noise=0.03
            )

    dr = NnDriver(
        data_source=ds,
        model_class=Model,
        future_count=15,
        verbose=True
        )
    
    dr.train(data_count=500, end_index=len(data) - dr.future_count)

    testing_end_index = training_end_index + 50

    dr.context_seq.data_source.data = SinusDs.Sinus(
        noise=0.03, stop=testing_end_index)
    dr.show_action(end_index=testing_end_index)

def main():  
    test()
    
if __name__ == "__main__":
    main()  


 