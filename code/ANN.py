import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from math import *
from operator import itemgetter
import os

class ANN(nn.Module):
    """
    MLP network
    """
    def init(self):
        """
        Setup for network
        Input:
        Output:
        """
        # network parameters
        self.num_observations = 24
        self.num_output = 1
        self.num_hidden_units = [8] # 8 neurons
        self.batch_size = 32 # data batches
        self.epochs = 30 # number of training iterations
        self.lr = 1e-04 # learning rate
        self.seed = 0
        self.rand_generator = np.random.RandomState(self.seed)

        # cuda device (GPU activation)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        # set random seed for each run
        self.rand_generator = np.random.RandomState(self.seed)

        # mean squared error loss function
        self.loss_func = nn.MSELoss()

        # initialise ANN networks
        self.network = nn.Sequential(
            nn.Linear(self.num_observations, self.num_hidden_units[0]),
            nn.Linear(self.num_hidden_units[0], self.num_output)
        ).to(self.device)

        # initialise model losses
        self.loss = None

        # define optimiser
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # decaying learning rate
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=1, gamma=0.1)

        # define network modes
        self.network.train()

    def train(self, input, output, wandb):
        """
        Network training
        Input:
            + input (tensor)
            + output (tensor)
            + wandb (object) -> metrics log
        Output:
            + loss_epochs (array)
            + accuracy_epochs (array)
        """
        self.loss_val = 0.0 # loss value init
        self.acc_val = 0.0 # accuracy value init
        self.loss_epochs = [] # epochs loss array (averaged)
        self.accuracy_epochs = [] # epochs accuracy array (averaged)
        for i in tqdm(range(self.epochs)): # iteration over dataset
            for j in range(len(input) // self.batch_size): # modify weights every batch of data
                # get input data batch
                idx = self.rand_generator.choice(len(input), size=self.batch_size, replace=False)
                batch = torch.from_numpy(np.array(itemgetter(*idx)(input.numpy()))).to(self.device)
                targets = torch.from_numpy(np.array(itemgetter(*idx)(output.numpy()))).to(self.device).float() # batch of ground truth data

                values = self.network(batch.float()) # feed batch forward
                self.optimiser.zero_grad() # reset gradients from optimiser
                self.loss = self.loss_func(values, targets) # compute loss
                self.loss.backward() # compute gradients
                self.optimiser.step() # backpropagation with ADAM optimiser

                self.loss_val += self.loss.detach().item() # update loss value
                self.acc_val += sqrt(self.loss.detach().item()) # Mean Squared Root Error

            self.loss_epochs.append(self.loss_val / (len(input)//self.batch_size)) # total epoch loss / number of data batches
            self.accuracy_epochs.append(self.acc_val / (len(input)//self.batch_size)) # mean absolute percentage error
            # self.scheduler.step() # learning rate decay every epoch

            # metrics log
            wandb.log({"Loss": self.loss_epochs[-1],
                "Accuracy (RMSE)": self.accuracy_epochs[-1],
                "Epochs": i+1,})

            # reset loss and accuracy
            self.loss_val = 0.0
            self.acc_val = 0.0

        # save model weights and metrics
        self.save('training')

    def test(self, input, output):
        """
        Testing
        Input:
            + input (tensor)
            + output (tensor)
        Output:
        """
        values = self.network(input.float()) # feedforward
        acc = sqrt(self.loss_func(values, output).detach().item())
        print(f"Accuracy (RMSE): {acc}%")

        fig = plt.figure("Forecast")
        plt.plot(values.squeeze(1).to("cpu").detach().numpy(), label="Predicted")
        plt.plot(output.squeeze(1).to("cpu").detach().numpy(), label="Ground Truth")
        plt.legend()
        plt.show()

    def prediction(self):
        """
        Predict data
        Input:
        Output:
        """
        self.network.eval() # network evaluation mode
        predictions = [] # initialise predictions

        # take last 24 rows of data from data.xlsx
        eval = torch.tensor([
        714.28497788115, 717.995463676785, 720.729432489218,
        721.718681997713, 721.11468954508, 725.286956896153,
        734.69056362186, 754.392652083173, 772.063389888152,
        781.329137146009, 794.006426501372, 813.111751526669,
        813.637445165078, 807.379389400738, 774.675210172325,
        751.198207847083, 740.272594458792, 712.851151081395,
        700.342408562234, 682.001337736434, 653.687926461454,
        645.29330201726, 619.618753357197, 594.570299960462
        ]).to(self.device)

        for i in range(40):
            val = self.network(eval) # prediction
            predictions.append(val.squeeze(0).detach().to("cpu")) # append results
            eval = torch.cat((eval, val), 0) # add prediction to input tensor
            eval = eval[-24:] # take last 24 input elements

        # plot predictions
        fig = plt.figure("Predictions")
        plt.plot(predictions, label="Predicted")
        plt.legend()
        plt.show()

        # save predictions to txt
        with open('./data/predictions.txt', 'w') as f:
            for i in predictions:
                f.write(f"{i}\n")

    def save(self, mode):
        """
        Used to save model weights and metrics
        Input:
            + mode (string)
        Output:
        """
        print("Saving weights and metrics")
        torch.save({
        "model_sate_dict": self.network.state_dict(),
        "optimiser_state_dict": self.optimiser.state_dict(),
        "loss": self.loss_epochs,
        "accuracy": self.accuracy_epochs}, f'./weights/{mode}Weights.tar')
        print("Model saved successfully")

    def load(self, mode):
        print("Loading model")
        checkpoint = torch.load(f'./weights/{mode}Weights.tar')
        self.network.load_state_dict(checkpoint["model_sate_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        print("Weights loaded successfully")
