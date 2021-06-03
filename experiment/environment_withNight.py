### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import gym
import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import __init__ as init
import weekCalendar
import humanSimulator_withNight
from random import randrange

class HumanEnv(gym.Env):
    """
    Description:
        A simulator is a human, which decide how this human perform PA after receiving a notification.
        Goal: deliver limited number of notifications to maxmim the total frequency of PA in a episode.
    Observation/state:
        Type: Box(10)
        Num	Observation                     Min        Max
        0	The notification left           0          __init__.max_notification => normalized into (0-1)
        1	Decision points from last run   0          __init__.max_decsionPerWeek - 1 => normalized into (0-1)
        2   Decision points from last notification 0   __init__.max_decsionPerWeek - 1 => normalized into (0-1)
        3   Date in a week                  0          6 -> one hot encoding
        4   Index in a day                  1          24 -> continous
        5   Temperature                     -10        36 -> continous
        6   Weather type                    1          8
        7   Wind type                       1          4
        8   Humidity type                   1          3 => normalized into (0-1)

    Actions:
        Type: Discrete(2)
        Num	Action
        0	not send a notification
        1	send a notification

    Reward:
        Every step, if notification is still available, actions have two options; otherwise, always action 1 will be taken
        Every step, if a run is performed, reward is 1.
    Episode Termination:
        Episode length is greater than MaxDecsionPerDay * 7 days
    Solved Requirements:
        Considered solved when the goal achieved continuously for the last 10 episode.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    # define parameters needed in the environment
    def __init__(self, num_episode, df):
        """
        Description:
            initialize the environment at the start of a run.
        Args:
            random_index (int): a random index of starting point in real data, each run has one particular start point.
            calendars (np.array): An array of object weekMatrix (weekCalendar), each calendar includes all info in a week read from data randomly.
            current_episode (int): the index of current calendar in calendars.
            current_calendar (object weekMatrix): The current calendar object.
            current_index (int): the index of step in the current calendar (index in a week).
            current_state (np.array): An array of current observation, ['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType'].
            last_run_index (int): the index of last run in this calendar.
            last_notification_index (int): the index of last notification in this calendar.
            steps_beyond_done (bool): whether this episode is done, True = done, False =  not done.
            action_space: spaces.Discrete(2)
            observation_space: spaces.Box(9)
            human (object dataGenerator.Human): The human simulator can decide whether run or not.
            init_memory (float): the initial memory randomly generated, which needed to be saved for re-run the experiment

        Return: initial observation state (np.array)
        """

        self.random_index = randrange(1800) * init.max_decsionPerDay
        self.calendars = self.generateCalendar(num_episode, self.random_index, df)  # generate all calendars from data
        self.current_episode = None
        self.current_index = None
        self.current_state = None
        self.last_run_index = None
        self.last_notification_index = None
        self.seed()
        self.viewer = None
        self.steps_beyond_done = None
        self.human = humanSimulator_withNight.Human()
        self.init_memory = self.human.getMemory()
        # self.goal_threshold = 3   # the total goal (frequency of run) in one episode

        # 0,1 represent send, not send.
        self.action_space = spaces.Discrete(2)

        # observation = ['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
        ## 'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType'].
        low = np.array([
            0,
            0,
            0,
            0,
            1,
            -10,
            1,
            1,
            1])
        high = np.array([
            init.max_notification,
            init.max_decsionPerWeek - 1,
            init.max_decsionPerWeek - 1,
            6,
            24,
            36,
            8,
            5,
            3])
        self.observation_space = spaces.Box(low, high, dtype=np.uint8)

    def getRandom_index(self):
        return self.random_index

    def getInit_memory(self):
        return self.init_memory

    def getCalendars(self):
        return self.calendars

    ##
    # Generate n calendars from data
    ##
    def generateCalendar(self, num, start_index, df):

        calendars = np.empty((0, 0), dtype='object')
        for i in range(num):
            # add one weekCalendar in np.array
            calendars = np.append(calendars, weekCalendar.weekMatrix(init.dayOfWeek, init.max_decsionPerDay, start_index, i, df))
        return calendars

    ##
    # produce a random seed, return np_random object to randomlize the actions
    ##
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ##
    # Reset/initialize the environment's state at the start of each episode.
    # @Returns observation.
    # observation = array([external condition = 1 / 0, notification left = notification_total,
    # decision from last run = 0,  total index = 0, index in a day = 0])
    ##
    def resetCalendar(self, index_episode):
        """
        Description:
            Read the calender info of each episode, as well as the environment's state at the start of current episode.
        Args:
            index_episode (int): An index of the current episode, used for get the context information from data.

        Info:
            state = np.array(['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType'])

        Return: initial observation state (np.array)
        """
        # read the current week calendar at the start of each episode
        self.current_episode = index_episode
        self.current_index = 0

        # when it is the start of a run for a new algorithm
        if self.current_episode == 0:
            self.last_run_index = None
            self.last_notification_index = None
            self.resetOneCalendar()

        if self.last_run_index is None:  # set there is no run ever, initialization as urge = 1
            self.last_run_index = - init.hour_to_urge - 1
        else:  # continue from last calendar
            self.last_run_index = self.last_run_index - init.max_decsionPerWeek

        if self.last_notification_index is None:
            # set there is no notification ever, initialization based on random initialized memory
            self.last_notification_index = - math.log(self.human.getMemory(), init.memory_scale)
        else:  # continue from last calendar
            self.last_notification_index = self.last_notification_index - init.max_decsionPerWeek

        # self.current_calendar.getGrid(0).getContext() =  np.array(['Weekday', 'Hour', 'Temperatuur', 'WeerType', 'WindType', 'LuchtvochtigheidType'])
        self.current_state = np.append(np.array([init.max_notification, self.current_index - self.last_run_index, self.current_index - self.last_notification_index]),
                               self.calendars[index_episode].getGrid(0).getContext())

        self.steps_beyond_done = False
        # do not reset human: not self.human = dataGenerator.Human()
        return self.current_state

    def resetOneCalendar(self):
        """
            Description:
                Reset attributes in each calender into default.
        """
        for calendar in self.calendars:
            calendar.setTotalRun(0)
            calendar.setNotificationLeft(init.max_notification)
            calendar.setTotalReward(0.0)

    def step(self, action):
        """
        Description:
            Step the environment using the chosen action by one timestep.
        Args:
            action (int): 1 = sent, 0 = not sent
        Info:
            state = np.array(['Notification_left', 'Time_from_lastRun', 'Time_from_lastNotifi',
                'weekday', 'hour', 'Temperatuur', ''WeerType', 'WindType', 'LuchtvochtigheidType'])
        Return: state (np.array), reward (float), done, info
                """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        index = self.current_index
        current_state = self.current_state

        """update the probability of send notification based on the current policy"""

        # if send notification
        if action == 1:
            # update the notification info in corresponding decision point object
            self.calendars[self.current_episode].getGrid(index).setNotification(True)

            # update the notification_left in corresponding calendar object
            self.calendars[self.current_episode].setNotificationLeft(self.calendars[self.current_episode].getNotificationLeft() - 1)

            # update the last_run_index in this environment object
            self.last_notification_index = index
        else:
            # update the notification info in corresponding decision point object
            self.calendars[self.current_episode].getGrid(index).setNotification(False)

        """given action and current_state, the user decide to run or not"""
        run, prob, weather_prob = self.human.isRun(action, current_state, index)
        # update the run_prob and weather_prob info in corresponding decision point object
        self.calendars[self.current_episode].getGrid(index).setProb(prob)
        self.calendars[self.current_episode].getGrid(index).setWeatherProb(weather_prob)

        if run:   # if run:
            reward = init.reward

            # update the run info in corresponding decision point object
            self.calendars[self.current_episode].getGrid(index).setRun(True)

            # update the the number of total in corresponding calendar object
            self.calendars[self.current_episode].setTotalRun(self.calendars[self.current_episode].getTotalRun() + 1)

            # update the total reward in corresponding calendar object
            self.calendars[self.current_episode].setTotalReward(self.calendars[self.current_episode].getTotalReward() + 1.0)

            # update the last_run_index in this environment object
            self.last_run_index = index

        else:   # if not run
            reward = 0.0

            # update the run info in corresponding decision point object
            self.calendars[self.current_episode].getGrid(index).setRun(False)

        # return all information of calendar in 'info', info is a dict
        info = {"calendars": self.calendars, "notification": self.calendars[self.current_episode].getNotificationLeft()}

        # check whether next state is available
        done = self.isEnd()

        if done:
            #print ("This episode is done.")
            # return
            return self.current_state, reward, done, info

        else:

            """update the attributes in this environment object"""
            self.current_index = index + 1

            """If we consider the night break"""
            if self.current_index % init.max_decsionPerDay == 0:
                self.current_state = np.append(
                    np.array([self.calendars[self.current_episode].getNotificationLeft(),
                              self.current_index - self.last_run_index + 8,
                              self.current_index - self.last_notification_index + 8]),
                              self.calendars[self.current_episode].getGrid(self.current_index).getContext())
            else:
                self.current_state = np.append(
                    np.array([self.calendars[self.current_episode].getNotificationLeft(),
                              self.current_index - self.last_run_index,
                              self.current_index - self.last_notification_index]),
                              self.calendars[self.current_episode].getGrid(self.current_index).getContext())
            # return
            return self.current_state, reward, done, info

    ##
    # check whether this episode is done
    ##
    def isEnd(self):
        if self.current_index == init.max_decsionPerWeek - 1:
                return True
        return False

    ##
    # visualize the environment
    ##
    def render(self, mode='human', close=False):
        return

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
