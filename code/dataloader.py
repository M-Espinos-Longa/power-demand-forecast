import pandas as pd
import math
import torch
import matplotlib.pyplot as plt

from datetime import datetime

class Dataloader(object):
    """
    Converts panda DataFrame into tensors
    """
    def __init__(self, data):
        """
        Input:
            + data (panda DataFrame)
        Output:
        """
        # copy data and shuffle
        self.raw_data = data

        self.look_back = 24 # input data entries

    def dataset(self):
        """
        Creates dataset
        Input:
        Output:
        """
        print("Generating data")
        self.trainIdata = torch.tensor([])
        self.testIdata = torch.tensor([])

        # training dataset
        for i in range(len(self.raw_data) - self.look_back):
            j = i - len(self.trainIdata) # testing tensor index

            traintensor = torch.tensor([])
            testtensor = torch.tensor([])

            # data edition
            row = self.raw_data.iloc[i:i+self.look_back+1] # get row
            vals = row[1] # column 1

            # add to training data
            if i < round(len(self.raw_data) * 0.9 - self.look_back):
                if i == 0:
                    for mb in range(i,i+self.look_back+1):
                        if mb < i+self.look_back:
                            self.trainIdata = torch.cat((self.trainIdata, torch.tensor([vals[mb]])), 0)
                        else:
                            self.trainIdata = self.trainIdata.unsqueeze(0)
                            self.trainOdata = torch.tensor([vals[mb]]).unsqueeze(0)
                else:
                    for mb in range(i,i+self.look_back+1):
                        if mb < i+self.look_back:
                            traintensor = torch.cat((traintensor, torch.tensor([vals[mb]])), 0)
                        else:
                            self.trainIdata = torch.cat((self.trainIdata, traintensor.unsqueeze(0)), 0)
                            self.trainOdata = torch.cat((self.trainOdata, torch.tensor([vals[mb]]).unsqueeze(0)), 0)

            # add to test data
            else:
                if i == round(len(self.raw_data) * 0.9 - self.look_back):
                    for mb in range(i,i+self.look_back+1):
                        if mb < i+self.look_back:
                            self.testIdata = torch.cat((self.testIdata, torch.tensor([vals[mb]])), 0)
                        else:
                            self.testIdata = self.testIdata.unsqueeze(0)
                            self.testOdata = torch.tensor([vals[mb]]).unsqueeze(0)
                else:
                    for mb in range(i,i+self.look_back+1):
                        if mb < i+self.look_back:
                            testtensor = torch.cat((testtensor, torch.tensor([vals[mb]])), 0)
                        else:
                            self.testIdata = torch.cat((self.testIdata, testtensor.unsqueeze(0)), 0)
                            self.testOdata = torch.cat((self.testOdata, torch.tensor([vals[mb]]).unsqueeze(0)), 0)

    def save(self):
        """
        Save tensors to external files
        Input:
        Output:
        """
        print("Saving datasets")
        # save training dataset
        torch.save(self.trainIdata, './data/trainInput.pt')
        torch.save(self.trainOdata, './data/trainOutput.pt')

        # save testing dataset
        torch.save(self.testIdata, './data/testInput.pt')
        torch.save(self.testOdata, './data/testOutput.pt')
        print("Datsets saved successfully")
