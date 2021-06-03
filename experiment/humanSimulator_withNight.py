### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import numpy as np
import pandas as pd
import pickle
import random
from pathlib import Path
import __init__ as init
import os

""""Consider the influence of night break"""

class Human(object):
    """ The human object, representing a person can decide run or not run.

        Args:
            memory (float): personal memory level of running.
            urge (float): mental motivation to run.
            prob (float): a parameter to balance the real human mental and assumptions/
    """

    def __init__(self):
        self.memory = init.memory_scale ** random.randrange(init.hour_to_forget)  # random initialize
        self.urge = 1   # start from 1
        self.prob = init.prob_run

    def getMemory(self):
        return self.memory

    ##
    # decision whether run or not run
    ##
    def isRun(self, action, state, index):
        """
        :param action: 1 is send notification, 0 is not.
        :param state: np.array(['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                          'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType'])
        :return: Boolean True = run, False = not run

        """
        prob, weather_prob = self.computeProb(action, state, index)

        if prob > 1.0:
            print "The probability of running is too high."
            return True, 1.0, weather_prob
        else:
            return np.random.choice([True, False], 1, p=[prob, 1 - prob]), prob, weather_prob


    def computeProb(self, action, state, index):
        """Compute the probablity of run. P(Rt| At-1, Rt-1, Mt-1, Nt, Ct)

        Args:
            action (bool): send notification or not, True = sent, False =  not sent.
            state (np.array): ['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                          'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType']
            last_run (bool): run or not in the last time step, True = run, False =  not run.

        Returns:
            prob (float): probability of running, in [0,1].
        """

        # get current urge value based on last_run and last_urge
        if state[1] == 1:  # if time_from_lastRun == 1
            self.urge = 0.001
        else:
            self.urge = self.urge + init.urge_scale
            if self.urge > 1:
                self.urge = 1.0

        # when it is the first hour in a day, update memory and urge
        if index % init.max_decsionPerDay == 0:
            self.memory = self.memory * (init.memory_scale ** 12)
            self.urge = self.urge + init.urge_scale * 12
            if self.urge > 1.0:
                self.urge = 1.0

        # get current memory value based on action and last_memory
        if action == 1:
            self.memory = 1.0
        else:
            self.memory = self.memory * init.memory_scale

        weather_prob = self.getProb(state[3:9])
        return self.memory * self.urge * weather_prob * self.prob, weather_prob


    def getProb(self, state):
        """Compute the probability P(Ct | Rt = 1) / P(Ct) from data.

            data saved in "/Users/Shihan/Desktop/Experiments/PAUL/rl/mylaps/weekday.csv"
                        "/Users/Shihan/Desktop/Experiments/PAUL/rl/mylaps/model/"
                        "/Users/Shihan/Desktop/Experiments/PAUL/rl/knmi/model/"
            Args:
                context (np.array): An array of context info ['Weekday', 'Hour', 'Temperatuur', 'WeerType', 'WindType', 'LuchtvochtigheidType'].

            Returns:
                prob (float): a probability in [0, ?) can be bigger than 1. !!!
            """

        inpath = init.dict

        #weekday = context[:, 1][0]
        #name = str(tuple(context[:, [3,4,5]][0]))
        #data = context[:, [0,1]]

        weekday = state[0]
        name = str(tuple(state[3:])).replace('.0', '')
        index = np.array([False, True, True, False, False, False])
        data = state[index]

        f_weekday = inpath + "mylaps/weekday.csv"
        df = pd.read_csv(f_weekday, encoding='utf-8-sig')

        prob_weekday = df.loc[df['Weekday'] == weekday, 'num'].iloc[0]

        f_knmi = inpath + "knmi/model/" + name + '.csv'
        p_knmi = os.path.expanduser(f_knmi)
        f_run = inpath + "mylaps/model/" + name + '.csv'
        p_run = os.path.expanduser(f_run)

        if os.path.exists(p_knmi):
            model = pickle.load(open(p_knmi, 'rb'))
            log_prob_knmi = model.score(data.reshape(1,-1))

            if os.path.exists(p_run):
                model = pickle.load(open(p_run, 'rb'))
                log_prob_run = model.score(data.reshape(1,-1))
            else:
                return 0.001
        else:
            return 1.0

        prob = prob_weekday * 7 * np.exp(log_prob_run - log_prob_knmi)

        return prob

