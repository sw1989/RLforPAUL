### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import numpy as np
import pandas as pd
import csv
import os
import __init__ as init

"""" Some functions to save information during the running of agents """


def saveInternal(calendar, i_run, i_episode, name):
    """
    Save the detailed info of a week in files.
    """

    file = init.dict + "finalresult/"
    path = os.path.expanduser(file + name + "_test/")

    data = np.asarray(calendar.returnCalendar())
    np.savetxt(path + str(i_run) + "_" + str(i_episode) + ".csv", data, delimiter=",", fmt='%s')
    
    print ("Save detailed info of a week in files done!")

def saveInternalLearning(calendar, i_run, i_episode, name):
    """
        Save the learning detailed info of a week in files (not all episodes).
    """
    
    file = init.dict + "finalresult/"
    path = os.path.expanduser(file + name + "/")
    
    data = np.asarray(calendar.returnCalendar())
    
    # save every 50 episode or the end episode
    if i_episode % 50 == 0 or i_episode == init.args.num_episodes - 1:
        np.savetxt(path + str(i_run) + "_" + str(i_episode) + ".csv", data, delimiter=",", fmt='%s')
        print ("Save learning detailed info of a week in files done!")

def saveParameter(env, agent, name, i_run):
    """
        Save the parameters that affects the randomness of simulation

        Args:
            index_in_data (int): the index of starting point in our environment data = env.getRandom_index()
            init_memory (int): the randomly initialized memory = env.getInit_memory()
            random_notifications_list (List of arrays): a list of index representing when a notification is sent,
            which was randomly generated for 'random_week algorithm' or 'random_day algorithm'
    """
    file = init.dict + "finalresult/"
    path = os.path.expanduser(file + name + "_policy/")

    index_in_data = env.getRandom_index()
    init_memory = env.getInit_memory()

    np.savetxt(path + str(i_run) + "_parameter.csv", np.array([index_in_data, init_memory]), delimiter=",", fmt='%s')
    data = np.asarray(agent.getSave())
    np.savetxt(path + str(i_run) + ".csv", data, delimiter=",", fmt='%s')

    print ("Random policy of one episode in one run saved.")

def savePolicy(agent, name, i_run, i_episode):
    """
        Save the parameters of the neural network in file via pytorch
    """

    file = init.dict + "finalresult/"
    path = os.path.expanduser(file + name + "_policy/")

    if i_episode % 50 == 0 or i_episode == init.args.num_episodes - 1:
        agent.save(path + str(i_run) + "_" + str(i_episode) + ".csv")

        print ("Learned policy of one episode in one run saved.")


def saveTofile(data, name, i_run):
    """
        Save data into file
    """

    """data = a list of rewards"""
    #"~/Workspace/run/result/"
    path = os.path.expanduser(init.dict + "finalresult/")
    with open(path + name + '_' + str(i_run) + '.csv', 'wb') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerow(data)

    print ("Reward of one run saved in file.")


