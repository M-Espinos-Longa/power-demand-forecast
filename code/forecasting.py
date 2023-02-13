import pandas as pd
from datetime import datetime
import wandb
import torch
from math import *
import os

from ANN import *
from dataloader import *

os.chdir("..") # level up directory

# initialise training log data
wandb.init(project="Forecasting", name=f"Training",
    notes=f"hu=[8], \
        lr=1e-04, \
        batch_size=32, \
        activation_func=None, \
        epochs=30, \
        look_back=24, \
        entity=wandbID") # fill in with your wandb ID (you will need to sign up at https://wandb.ai/site)

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
training_data_input = torch.load('./data/trainInput.pt', map_location=torch.device(net.device))
training_data_output = torch.load('./data/trainOutput.pt', map_location=torch.device(net.device))
testing_data_input = torch.load('./data/testInput.pt', map_location=torch.device(net.device))
testing_data_output = torch.load('./data/testOutput.pt', map_location=torch.device(net.device))

# train and test model
#net.train(training_data_input, training_data_output, wandb)
net.load("training") # (UNCOMMENT IF WEIGHTS AVAILABLE)
#net.test(testing_data_input, testing_data_output)

# prediction
net.prediction()
