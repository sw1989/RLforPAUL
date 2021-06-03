### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import numpy as np
import __init__ as init


class decisionPoint(object):

    """ The decision point object, representing one hour in calendar

        Args:
            x (int): the index of x-axis = date in a week, from 0 to 6.
            y (int): the index of y-axis, [0, 15].
            index (int): the index of this decision point in all calendars.
            context (np.array): An array of context info ['Weekday', 'Hour', 'Temperatuur', 'WeerType', 'WindType', 'LuchtvochtigheidType']
                weekday (int): day in a week = self.x + 1, from Sunday to Saturday.
                hour (int): hour in a day = self.y + 6.
                temperature (int): temperature at current decision point [-10, 36].
                weather (int): weather type from 1 to 8 in [sunny, cloudy, half cloudy, rainy, storm, snow, hail, mist].
                wind (int): wind type from 1 to 5 in [windless -> storm]
                humidity (int): humidity type from 1 to 3 in [low, moderate, high]
            is_notification (bool): whether a notification was sent in this decision point, send = True, not send = False.
            is_run (bool): whether the user run in this decision point, run = True, not run = False.
    """

    def __init__(self, episode, x, y, context):
        self.x = x
        self.y = y
        self.index = episode * init.max_decsionPerWeek + init.max_decsionPerDay * x + y
        self.context = context
        self.is_run = None
        self.is_notification = None
        self.run_prob = None # the probability of run
        self.weather_prob = None # the P(C|R) / P(C)

    def getContext(self):
        return self.context

    def setProb(self, prob):
        self.run_prob = prob

    def getProb(self):
        return self.run_prob

    def setWeatherProb(self, prob):
        self.weather_prob = prob

    def getWeatherProb(self):
        return self.weather_prob

    def setNotification(self, notification):
        self.is_notification = notification

    def getNotification(self):
        return self.is_notification

    def setRun(self, run):
        self.is_run = run

    def getRun(self):
        return self.is_run

    # return the information of this decision point as an numpy.array
    def returnInfo(self):
        # index, x, y, notification, run, run_prob, weather_prob, context [weekday, hour, temperature, weatherm wind, humidity]
        arr = np.array([self.index, self.x, self.y, init.switchBinary(self.is_notification),
                        init.switchBinary(self.is_run), self.run_prob, self.weather_prob])
        return np.concatenate([arr, self.context])
