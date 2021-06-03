### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import torch
import numpy as np
import random
import __init__ as init
from torch.autograd import Variable

class Random:
    def __init__(self):
        """ The Random object, send a certain number of notifications in a week randomly.

                Args:
                    notification_index (np.array): an array of index representing the decision points to send a notification.
                    notification_index_list (List of np.array): a list of all historical notification_index in this run
            """
        self.notification_index = None
        self.notification_index_list = []


    def reset(self):
        """reset init.max_notification number of notification randomly"""
        self.notification_index = np.array(random.sample(range(0, init.max_decsionPerWeek), init.max_notification))

    def saveUpdate(self):
        """save the current notification_index into a list"""
        array = self.notification_index
        self.notification_index_list.append(array)

    def getSave(self):
        """return a list of array"""
        return self.notification_index_list

    ##
    # select the action for this index
    ##
    def select_action(self, index, state):

        if index in self.notification_index:
            return torch.tensor([1], dtype=torch.int32), torch.tensor([[0.0]], dtype=torch.float), torch.tensor([[0.0]], dtype=torch.float)
        else:
            return torch.tensor([0], dtype=torch.int32), torch.tensor([[0.0]], dtype=torch.float), torch.tensor([[0.0]], dtype=torch.float)

