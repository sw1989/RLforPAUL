### This is the python code for simulation experiments in Wang, Zhang, Krose, & van Hoof, 2021
import __init__ as init
import numpy as np
import saveInfo
import os
import torch

"""" Some functions to control different running environments of algorithms (differ training and testing; with and without restriction) """

# run without environment restriction, the algorithm is learning
def run_learn(agent, name, env, i_run, baseline, num_episode):
    """
        the run function for learning algorithms (reinforce_agent and reinforce_restrict_agent)
    """
    episode_rewards = []  # the total reward at each timestamp in one episode
    # average_rewards = []    # the average reward for episodes by now

    run_match_notification_reward = []  # the number of run after a notification in each episode
    # ave_run_match_notification = []  # the average number of run after a notification for episodes by now
    notification_left = [] # the number of notification left at the end of each episode
    wrong_notification = [] # the number of wrong notification in each episode
    extra_wrong_notification = [] # the number of extra wrong notification (notification right after a run) in each episode

    """" training: loop the episodes """
    # for each episode, update the parameter
    for i_episode in range(num_episode):

        # env.reset(): Reset the environment as the current episode. Return the first observation
        state = env.resetCalendar(i_episode)

        log_probs = []
        rewards = []  # the reward at each step
        rewards_notifi = []  # the reward at each step before the notification was sent out
        current_probs = []  # the probability of sending notification at each step

        """" training: loop the steps in each episode """
        for t in range(init.args.num_steps):
            action, log_prob, current_prob = agent.select_action(state)
            ## action = action.cpu()

            # env.step(self, action): Step the environment using the chosen action by one timestep.
            # Return observation (np.array), reward (float), done (boolean), info (dict) """
            state, reward, done, info = env.step(action.numpy()[0])

            # save all the rewards in this episode
            current_probs.append((t, current_prob))
            log_probs.append(log_prob)
            rewards.append(reward)

            # append rewards only before notification was sent out
            if info['notification'] > 0 or action.numpy()[0] == 1:
                rewards_notifi.append(reward)

            # once this episode finished
            if done:
                print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

                # same the sum reward of each episode for plot
                episode_rewards.append(np.sum(rewards))
                # average_rewards.append(sum(episode_rewards) / (len(episode_rewards) + 0.0))
                # average_rewards_window = meanInWindow(episode_rewards, 20)

                run_match_notification_reward.append(getReward(info['calendars'], i_episode))
                # ave_run_match_notification.append(sum(run_match_notification_reward) / (len(run_match_notification_reward) + 0.0))
                
                notification_left.append(getNotificationLeft(info['calendars'], i_episode))
                
                wrong, extra_wrong = getWrongNotification(info['calendars'], i_episode)
                wrong_notification.append(wrong)
                extra_wrong_notification.append(extra_wrong)

                """save detailed information of the current week into files"""
                saveInfo.saveInternalLearning(info['calendars'][i_episode], i_run, i_episode, name)

                break

        # update the policy at the end of episode
        agent.finish_episode(rewards, init.args.gamma, log_probs, 3.5, [], i_episode)
        saveInfo.savePolicy(agent, name, i_run, i_episode)

    print ("One learning run done!")

    return episode_rewards, notification_left, wrong_notification, extra_wrong_notification, agent

# run with environment restriction, the algorithm is learning
def run_restrict(agent, name, env, i_run, baseline, start_episode, num_episode):
    """
        the run function for learning algorithms (reinforce_agent and reinforce_restrict_agent)
        """
    episode_rewards = []  # the total reward at each timestamp in one episode
    # average_rewards = []    # the average reward for episodes by now
    
    run_match_notification_reward = []  # the number of run after a notification in each episode
    notification_left = [] # the number of notification left at the end of each episode
    wrong_notification = [] # the number of wrong notification in each episode
    extra_wrong_notification = [] # the number of extra wrong notification (notification right after a run) in each episode
    
    """" training: loop the episodes """
    # for each episode, update the parameter
    for i_episode in range(start_episode, num_episode):
        
        # env.reset(): Reset the environment as the current episode. Return the first observation
        state = env.resetCalendar(i_episode)
        
        log_probs = []
        rewards = []  # the reward at each step
        rewards_notifi = []  # the reward at each step before the notification was sent out
        current_probs = []  # the probability of sending notification at each step
        
        """" training: loop the steps in each episode """
        for t in range(init.args.num_steps):
            action, log_prob, current_prob = agent.select_action(state)
            ## action = action.cpu()
            
            # if no notification left, always do not send notification
            if state[0] == 0:
                action = torch.tensor([0], dtype=torch.int32)
            
            # env.step(self, action): Step the environment using the chosen action by one timestep.
            # Return observation (np.array), reward (float), done (boolean), info (dict) """
            state, reward, done, info = env.step(action.numpy()[0])
            
            # save all the rewards in this episode
            current_probs.append((t, current_prob))
            log_probs.append(log_prob)
            rewards.append(reward)
            
            # append rewards only before notification was sent out
            if info['notification'] > 0 or action.numpy()[0] == 1:
                rewards_notifi.append(reward)
        
            # once this episode finished
            if done:
                print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
                
                # same the sum reward of each episode for plot
                episode_rewards.append(np.sum(rewards))
                # average_rewards.append(sum(episode_rewards) / (len(episode_rewards) + 0.0))
                # average_rewards_window = meanInWindow(episode_rewards, 20)
                
                run_match_notification_reward.append(getReward(info['calendars'], i_episode))
                # ave_run_match_notification.append(sum(run_match_notification_reward) / (len(run_match_notification_reward) + 0.0))
                
                notification_left.append(getNotificationLeft(info['calendars'], i_episode))
                
                wrong, extra_wrong = getWrongNotification(info['calendars'], i_episode)
                wrong_notification.append(wrong)
                extra_wrong_notification.append(extra_wrong)
                
                """save detailed information of the current week into files"""
                saveInfo.saveInternalLearning(info['calendars'][i_episode], i_run, i_episode, name)
                
                break

        # update the policy at the end of episode
        agent.finish_episode(rewards, init.args.gamma, log_probs, 3.5, [], i_episode)
        saveInfo.savePolicy(agent, name, i_run, i_episode)

    print ("One learning run done!")

    return episode_rewards, notification_left, wrong_notification, extra_wrong_notification

# run with environment restriction, the algorithm is not learning any more
def run_test(agent, name, env, i_run):
    """
        the run function for learning algorithms (reinforce_agent and reinforce_restrict_agent)
    """
    episode_rewards = []  # the total reward at each timestamp in one episode
    run_match_notification_reward = []  # the number of run after a notification in each episode
    notification_left = [] # the number of notification left at the end of each episode
    wrong_notification = [] # the number of wrong notification in each episode
    extra_wrong_notification = [] # the number of extra wrong notification (notification right after a run) in each episode


    """" training: loop the episodes """
    # for each episode, update the parameter
    for i_episode in range(init.args.test_episodes - init.args.left_episodes, init.args.test_episodes):

        # env.reset(): Reset the environment as the current episode. Return the first observation
        state = env.resetCalendar(i_episode)

        log_probs = []
        rewards = []  # the reward at each step
        rewards_notifi = []  # the reward at each step before the notification was sent out
        current_probs = []  # the probability of sending notification at each step

        """" training: loop the steps in each episode """
        for t in range(init.args.num_steps):
            
            # if no notification left, always do not send notification
            if state[0] == 0:
                action = torch.tensor([0], dtype=torch.int32)
            else:
                action, log_prob, current_prob = agent.select_action(state)
                ## action = action.cpu()

            # env.step(self, action): Step the environment using the chosen action by one timestep.
            # Return observation (np.array), reward (float), done (boolean), info (dict) """
            state, reward, done, info = env.step(action.numpy()[0])

            # save all the rewards in this episode
            rewards.append(reward)

            # append rewards only before notification was sent out
            if info['notification'] > 0 or action.numpy()[0] == 1:
                rewards_notifi.append(reward)

            # once this episode finished
            if done:
                print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

                # same the sum reward of each episode for plot
                episode_rewards.append(np.sum(rewards))
                run_match_notification_reward.append(getReward(info['calendars'], i_episode))
                notification_left.append(getNotificationLeft(info['calendars'], i_episode))

                wrong, extra_wrong = getWrongNotification(info['calendars'], i_episode)
                wrong_notification.append(wrong)
                extra_wrong_notification.append(extra_wrong)

                """save detailed information of the current week into files"""
                #saveInfo.saveInternal(info['calendars'][i_episode], i_run, i_episode, name)
                saveInfo.saveInternalLearning(info['calendars'][i_episode], i_run, i_episode, name)

                break

    print ("One testing run done!")

    return episode_rewards, notification_left, wrong_notification, extra_wrong_notification

# run without environment restriction for rule-based algorithms
def run_random(agent, name, env, i_run):
    """
            the run function for random-setup algorithms (random_week and random_day)
    """

    episode_rewards = []  # the total reward at each timestamp in one episode
    # average_rewards = []    # the average reward for episodes by now

    run_match_notification_reward = []  # the number of run after a notification in each episode
    # ave_run_match_notification = []  # the average number of run after a notification for episodes by now
    notification_left = [] # the number of notification left at the end of each episode
    wrong_notification = [] # the number of wrong notification in each episode
    extra_wrong_notification = [] # the number of extra wrong notification (notification right after a run) in each episode

    """" training: loop the episodes """
    # for each episode, update the parameter
    for i_episode in range(init.args.test_episodes):

        agent.reset()
        agent.saveUpdate()

        # env.reset(): Reset the environment as the current episode. Return the first observation
        state = env.resetCalendar(i_episode)

        rewards = []  # the reward at each step
        # rewards_notifi = []  # the reward at each step before the notification was sent out

        """" training: loop the steps in each episode """
        for t in range(init.args.num_steps):
            action, log_prob, current_prob = agent.select_action(t, state)
            ## action = action.cpu()

            # env.step(self, action): Step the environment using the chosen action by one timestep.
            # Return observation (np.array), reward (float), done (boolean), info (dict) """
            state, reward, done, info = env.step(action.numpy()[0])

            # save all the rewards in this episode
            rewards.append(reward)

            # append rewards only before notification was sent out
            # if info['notification'] > 0 or action.numpy()[0] == 1:
            #    rewards_notifi.append(reward)

            # once this episode finished
            if done:
                print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

                # same the sum reward of each episode for plot
                episode_rewards.append(np.sum(rewards))
                # average_rewards.append(sum(episode_rewards) / (len(episode_rewards) + 0.0))
                # average_rewards_window = meanInWindow(episode_rewards, 20)

                run_match_notification_reward.append(getReward(info['calendars'], i_episode))
                # ave_run_match_notification.append(sum(run_match_notification_reward) / (len(run_match_notification_reward) + 0.0))

                notification_left.append(getNotificationLeft(info['calendars'], i_episode))
                wrong, extra_wrong = getWrongNotification(info['calendars'], i_episode)
                wrong_notification.append(wrong)
                extra_wrong_notification.append(extra_wrong)
                
                """save learning detailed information of the current week into files"""
                saveInfo.saveInternalLearning(info['calendars'][i_episode], i_run, i_episode, name)

                break

        """save random policy and initial parameters """
        #saveInfo.saveParameter(env, agent, name, i_run)

    print ("One random run done!")
    return episode_rewards, notification_left, wrong_notification, extra_wrong_notification


def getReward(calenders, i_episode):
    """
    Calculate the reward if people run after they receive the notification
    :param calenders: the whole calenders of an environment
    :param i_episode: the index of current week
    :return: how many notification was left & how many reward can get
    """

    reward = 0.0

    # add reward in if decision_point.isRun == decision_point.isNotification == True
    for index in range(init.max_decsionPerWeek):
        decision_point = calenders[i_episode].getGrid(index)

        if decision_point.getRun() and decision_point.getNotification():
            reward = reward + 1.0

    return reward

def getWrongNotification(calenders, i_episode):
    """
        Calculate how many of notifications were sent after a user has been run during the day.
        :param calenders: the whole calenders of an environment
        :param i_episode: the index of current week
        :return: how many notification was left & how many reward can get
        """
    
    wrong_notification = 0.0
    extra_wrong_notification = 0.0
    
    # add reward in if decision_point.isRun == decision_point.isNotification == True
    for index in range(init.max_decsionPerWeek):
        decision_point = calenders[i_episode].getGrid(index)
        
        if (index+1) % init.max_decsionPerDay != 0 and decision_point.getRun():
            # calculate which date it is now
            date = index / init.max_decsionPerDay
            # for all decision points after this run before next day, if there is a notification sent
            for next_index in range (index+1, (date+1) * init.max_decsionPerDay):
                if calenders[i_episode].getGrid(next_index).getNotification():
                    wrong_notification = wrong_notification + 1.0
        
            if calenders[i_episode].getGrid(index + 1).getNotification():
                    extra_wrong_notification = extra_wrong_notification + 1.0

    return wrong_notification, extra_wrong_notification


# return the notificatoin left in this episode
def getNotificationLeft(calenders, i_episode):
    return calenders[i_episode].getNotificationLeft()

# compute the mean of a window
def meanInWindow(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
