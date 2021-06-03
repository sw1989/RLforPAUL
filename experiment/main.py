### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import torch
import pandas as pd
import numpy as np
import os
import REINFORCE_win_baseline as REINFORCE_win_baseline
import RESTRICT_win_baseline as RESTRICT_win_baseline
import Learned as learned
import fixed_day as fixed_day
import random_week as random_week
import random_day as random_day
import environment_withNight
import __init__ as init
import run_save
import saveInfo
from torch.autograd import Variable
import torch.nn.utils as utils
import cProfile
import re

#from multiprocessing import Process, Manager, Value, Lock
import time

DISPLAY_REWARD_THRESHOLD = 100  # show visual window when reward > 400
RENDER = False
"""" set up the dictionary for saving """

# This is an example main function for running one simulation. You may set up your own code for running different experiments (and multiple jobs).
def oneEnvironment(i_run):

    """read Context info from documents and put them in df"""
    input = os.path.expanduser(init.dict + "knmi/knmi_weekday.csv")
    df = pd.read_csv(input, encoding='utf-8-sig').dropna()

    """" set up the training environment """
    env = environment_withNight.HumanEnv(init.args.num_episodes, df)

    # random seed
    env.seed(init.args.seed)
    torch.manual_seed(init.args.seed)
    np.random.seed(init.args.seed)

    # delete the wrap
    env = env.unwrapped

    """" set up the training agent """

    #Baseline agents
    
    #Random_day algorithm: send notification with maximal init.maxnotification/7 per day randomly
    agent_day = random_day.Random()
    raw_rewards_d, notification_left_d, wrong_n_d, extra_wrong_n_d = run_save.run_random(agent_day, 'day', env, i_run)
    saveInfo.saveTofile(raw_rewards_d, "day_train", i_run)
    saveInfo.saveTofile(notification_left_d, "day_train_notification", i_run)
    saveInfo.saveTofile(wrong_n_d, "day_train_wrong", i_run)
    saveInfo.saveTofile(extra_wrong_n_d, "day_train_extra_wrong", i_run)
    
    #Random_fix algorithm: send notification with maximal init.maxnotification/7 per day at fixed times
    agent_fix = fixed_day.Random()
    raw_rewards_f, notification_left_f, wrong_n_f, extra_wrong_n_f = run_save.run_random(agent_fix, 'fix', env, i_run)
    saveInfo.saveTofile(raw_rewards_f, "fix_train", i_run)
    saveInfo.saveTofile(notification_left_f, "fix_train_notification", i_run)
    saveInfo.saveTofile(wrong_n_f, "fix_train_wrong", i_run)
    saveInfo.saveTofile(extra_wrong_n_f, "fix_train_extra_wrong", i_run)

    #Random_week algorithm: send notification with maximal init.maxnotification randomly
    agent_week = random_week.Random()
    raw_rewards_w, notification_left_w, wrong_n_w, extra_wrong_n_w = run_save.run_random(agent_week, 'week', env, i_run)
    saveInfo.saveTofile(raw_rewards_w, "week_train", i_run)
    saveInfo.saveTofile(notification_left_w, "week_train_notification", i_run)
    saveInfo.saveTofile(wrong_n_w, "week_train_wrong", i_run)
    saveInfo.saveTofile(extra_wrong_n_w, "week_train_extra_wrong", i_run)

    baseline = np.mean(raw_rewards_w)
    #baseline = 3.5

    #RL agents
    # REINFORCE algorithm: send notification with no maximal notifications
    agent_baseline = REINFORCE_win_baseline.REINFORCE(init.args.hidden_size, 27, env.action_space, baseline)
    raw_rewards_rb_train, notification_left_rb_train, wrong_n_rb_train, extra_wrong_n_rb_train, agent_reinforce_learned = run_save.run_learn(agent_baseline, 'reinforce', env, i_run, baseline, init.args.num_episodes - init.args.left_episodes)
    raw_rewards_rb_test, notification_left_rb_test, wrong_n_rb_test, extra_wrong_n_rb_test = run_save.run_restrict(agent_reinforce_learned,'reinforce', env, i_run, baseline, init.args.test_episodes - init.args.left_episodes, init.args.test_episodes)
    saveInfo.saveTofile(raw_rewards_rb_train + raw_rewards_rb_test, "reinforce_train", i_run)
    saveInfo.saveTofile(notification_left_rb_train + notification_left_rb_test, "reinforce_train_notification", i_run)
    saveInfo.saveTofile(wrong_n_rb_train + wrong_n_rb_test, "reinforce_train_wrong", i_run)
    saveInfo.saveTofile(extra_wrong_n_rb_train + extra_wrong_n_rb_test, "reinforce_train_extra_wrong", i_run)

    # REINFORCE_restrict algorithm: send notification with maximal init.max_notification
    agent_restrict_win = RESTRICT_win_baseline.REINFORCE(init.args.hidden_size, 28, env.action_space, baseline)
    raw_rewards_rw_train, notification_left_rw_train, wrong_n_rw_train, extra_wrong_n_rw_train, agent_restrict_learned = run_save.run_learn(agent_restrict_win, 'restrict_win',env, i_run, baseline, init.args.num_episodes)
    saveInfo.saveTofile(raw_rewards_rw_train, "restrict_win_train", i_run)
    saveInfo.saveTofile(notification_left_rw_train, "restrict_win_notification", i_run)
    saveInfo.saveTofile(wrong_n_rw_train, "restrict_win_wrong", i_run)
    saveInfo.saveTofile(extra_wrong_n_rw_train, "restrict_win_extra_wrong", i_run)

    env.close()

