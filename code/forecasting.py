import pandas as pd
from datetime import datetime
import wandb
import torch
from math import *

from ANN import *
from dataloader import *

# initialise training log data
wandb.init(project="Forecasting", name=f"Training",
    notes=f"hu=[64], \
        lr=1e-03 (decay rate=0.1), \
        batch_size=32, \
        activation_func=Tanh, \
        epochs=100, \
        look_back=144, \
        entity=m-espinos-longa")

# define metrics
wandb.define_metric("Epochs")
wandb.define_metric("Loss", step_metric="Epochs")
wandb.define_metric("Accuracy", step_metric="Epochs")

# data management (AVOID IF DATA IS ALREADY GENERATED)
# data = pd.read_excel('data.xlsx', sheet_name='Sheet1', usecols=[0, 1],
#     skiprows=[0], header=None) # read data
# dl = Dataloader(data) # initialise dataloader
# dl.dataset() # generate dataset
# dl.save() # save datset

# model init
net = ANN()
net.init()

# load data (AVOID IF NO GENERATED DATA)
training_data_input = torch.load('trainInput.pt', map_location=torch.device("cpu"))
training_data_output = torch.load('trainOutput.pt', map_location=torch.device("cpu"))
testing_data_input = torch.load('testInput.pt', map_location=torch.device(net.device))
testing_data_output = torch.load('testOutput.pt', map_location=torch.device(net.device))

# train and test model
#net.train(training_data_input, training_data_output, wandb)
net.load("training")
net.test(testing_data_input, testing_data_output)
