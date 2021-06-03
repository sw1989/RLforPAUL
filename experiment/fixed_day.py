### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import torch
import numpy as np
import random
import __init__ as init

class Random:
    def __init__(self):
        """ The object, send a certain number of notifications in a day at fixed times.

                Args:
                    notification_hours (np.array): an array of index representing the hours to send notifications in a day.
            """
        self.notification_hours = None
        self.notification_hours_list = []

    def reset(self):
        # set up the fixed hours
        self.notification_hours = np.array([12, 16])

    def saveUpdate(self):
        """save the current notification_index into a list"""
        array = self.notification_hours
        self.notification_hours_list.append(array)

    def getSave(self):
        """return a list of array"""
        return self.notification_hours_list

    def select_action(self, index, state):

        # reset the notification_hours randomly everyday
        if index != 0 and index % init.max_decsionPerDay == 0:
            self.reset()

        if state[4] in self.notification_hours:
            return torch.tensor([1], dtype=torch.int32), torch.tensor([[0.0]], dtype=torch.float), torch.tensor([[0.0]], dtype=torch.float)
        else:
            return torch.tensor([0], dtype=torch.int32), torch.tensor([[0.0]], dtype=torch.float), torch.tensor([[0.0]], dtype=torch.float)
