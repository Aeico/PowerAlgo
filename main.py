# %%
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import homeenv
import model

Agent = model.Agent

Home_Environment = homeenv.Home_Environment

import json
import time

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def plot_env(x, charge, exchange, filename, should_show, lines=None):
    fig=plt.figure(figsize=(6*2.5, 3*2.5), dpi=200)

    x1=fig.add_subplot(111, label="1")
    x2=fig.add_subplot(111, label="2", frame_on=False)
    
    x1.plot(x, exchange, color="C0")
    x1.set_xlabel("Days", color="k")
    x1.set_ylabel("Exchange", color="C0")
    x1.tick_params(axis='x', colors="k")
    x1.tick_params(axis='y', colors="C0")

    x2.scatter(x, charge, color="C1")
    x2.set_ylabel("Charge", color="C1")
    x2.yaxis.set_label_position('right')
    x2.yaxis.tick_right()
    x2.tick_params(axis='x', colors="k")
    x2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    x1.set_ylim(-5,100)
    x2.set_ylim(-0.2,7)
    plt.savefig(filename)
    if(should_show == True):
        plt.show()

def plot_env_score(x, data, score, filename, should_show, lines=None):
    
    fig=plt.figure(figsize=(6*2.5, 3*2.5), dpi=200)

    x1=fig.add_subplot(111, label="1")
    x2=fig.add_subplot(111, label="2", frame_on=False)
    x1.plot(x, score, color="C2")
    x1.set_xlabel("Days", color="k")
    x1.set_ylabel("Score", color="C2")
    x1.tick_params(axis='x', colors="C2")
    x1.tick_params(axis='y', colors="C2")

    arr_mean_price = []
    temp = []
    for i in range(len(x)): #Average price of electricity
        for j in range(24):
            temp.append(data[j*(i+1)]['Value'])
        if(temp[0])== 0:
            arr_mean_price.append(0)
        arr_mean_price.append(np.mean(temp))
        temp = []
    
    x2.plot(x, arr_mean_price, color="C3")
    x2.set_ylabel("Mean Price", color="C3")
    x2.yaxis.set_label_position('right')
    x2.yaxis.tick_right()
    x2.tick_params(axis='x', colors="k")
    x2.tick_params(axis='y', colors="C3")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    if(should_show == True):
        plt.show()

def if_step_sell(days, step_period, show, data):
    #Home Environment

    env = Home_Environment(step_period,data)
    counts = [0,0,0]
    state = env.day_init()
    while env.day_index <= days:#((len(data)/24)):
        #if price below 33 and charge not full
        if state [0] > 40 or env.home_charge + env.home_discharge_rate_temp*env.step_period + env.solar_charge_rate_temp*env.step_period >= env.max_charge:
            state, reward, done = env.step(0)
        else: 
            state, reward, done = env.step(1)
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i+1 for i in range(days+1)]
    filename = 'ifstep.png'

    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)
    return np.round(env.total_sold_price/100 - env.total_bought_price/100,3)

def only_sell(days, step_period, show, data):
    env = Home_Environment(step_period,data)
    counts = [0,0,0]
    while env.day_index <= days:#((len(data)/24)):
        state, reward, done = env.step(0)
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i+1 for i in range(days+1)]
    filename = 'onlystepsell.png'

    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)
    return np.round(env.total_sold_price/100 - env.total_bought_price/100,3)

def only_buy(days, step_period, show, data):
    env = Home_Environment(step_period,data)
    counts = [0,0,0]
    days = days
    while env.day_index <= days:#((len(data)/24)):
        state, reward, done = env.step(1)
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i+1 for i in range(days+1)]
    filename = 'onlystepbuy.png'
    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)
    return np.round(env.total_sold_price/100 - env.total_bought_price/100,3)

def test_rewards(days, step_period, show, data):
    env = Home_Environment(step_period,data)
    counts = [0,0,0]
    days = days
    while env.day_index <= days:#((len(data)/24)):

        tot_diff = 0
        for i in range(4):
            new_diff = (env.data[int(i+(env.time/3600))]['Value'] \
                - env.data[int(i+1+(env.time/3600))]['Value'])
            new_diff = new_diff/(1+(i/10))
            tot_diff += new_diff

        if tot_diff > 2.2 \
                or env.home_charge + env.home_discharge_rate_temp*env.step_period + env.solar_charge_rate_temp*env.step_period >= env.max_charge:
            state, reward, done = env.step(0)
        else: 
            state, reward, done = env.step(1)
        
        if (env.day_index % 100 == 0 and env.time/3600 % 1200 == 1):
            print(f"Day: {env.day_index}")
            ampere_hour = env.home_charge/3600
            print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah | Reward was: {np.round(reward,3)}")
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i+1 for i in range(days+1)]
    
    filename = 'reward_test.png'
    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)

    return np.round(env.total_sold_price/100 - env.total_bought_price/100,3)

def deep_q_agent(days, step_period, show, data, save, save_path, load, load_path):
    env = Home_Environment(step_period,data)
    agent = Agent(gamma=0.99, epsilon=0.01, batch_size=32, n_actions=2, 
                eps_end=0.0001, eps_dec=5e-7, input_dims=[7], lr=0.0001,
                load_path=load_path, load=load)
    scores, eps_history = [],[]
    action_count = [0,0,0]

    for i in range(days):
        score = 0
        done = False
        observation = env.day_init()
        while not done:
            action = agent.choose_action(observation) #Calculate action or if greedy set it to random
            #CHANGE HERE FOR REALTIME
            observation_, reward, done = env.step(action) #Step forward (If realtime needs to be different)

            score += reward
            action_count[action] += 1
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        if i % 100 == 0:
            ampere_hour = env.home_charge/3600
            print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah | Reward was: {np.round(reward,3)}")
            print('day', i, 'score %.3f' % score,'average score %.3f' % avg_score,'epsilon %.3f' % agent.epsilon)
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i for i in range(days)]    
    filename = 'agent_performance.png'
    env.charge_at_time.append(env.home_charge/3600)
    env.exchange.append((env.total_sold_price-env.total_bought_price)/100)

    mean_scores = []
    for i in range (len(scores)):
        mean_scores.append(np.mean(scores[(-5*i):]))

    #plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)
    #plot_env_score(x, data, scores, 'agent_score.png',should_show=show)

    print(action_count)
    
    if save:
        torch.save(agent.Q_eval.state_dict(), save_path)
    
    return np.round(env.total_sold_price/100 - env.total_bought_price/100,3)

#All functions use (Days, Step_Period(60 = 1 min), Show graph)
if __name__ == "__main__":
    start_time = time.time()

    train = False #Train if true or Test if false
    override = False #If want custom data / days

    if train:
        file = open('2008till2022-10-15.json')
        days = 4000
    elif not train:
        file = open('2021till2022nov.json')
        days = 674
    if override:# Custom stuff here
        file = open('2021till2022nov.json')
        #file = open('virtualdata.json')
        days = 10

    data = json.load(file)

    save = False
    load = True
    save_path = "Q_EvalTest_FromOld.pth"
    load_path = "Q_EvalTest_FromOld.pth"

    step = 60*60

    model = ["if sell", "sell", "buy", "formula", "untrained", "trained"]
    profit = []

    print("\n-----------------------------------------------------------------------------------------------\n")
    print(f'Days in data: {int((len(data)/24) -1)}')#Prints amount of days in data given
    print("\n-----------------------------------------------------------------------------------------------\nIf Price > 0.40:")
    run = if_step_sell(days, step, False, data)
    profit.append(run)
    print("\n-----------------------------------------------------------------------------------------------\nOnly sell:")
    run = only_sell(days, step, False, data)
    profit.append(run)
    print("\n-----------------------------------------------------------------------------------------------\nOnly buy:")
    run = only_buy(days, step, False, data)
    profit.append(run)
    print("\n-----------------------------------------------------------------------------------------------\nDepending on current price compared to price in 1 and 2 hours")
    run = test_rewards(days, step, False, data)
    profit.append(run)
    print("\n-----------------------------------------------------------------------------------------------\nDQN with Adam Optimzer and Experience Replay, Training not loaded")
    run = deep_q_agent(days, step, False, data, save=False, save_path=save_path, load=False, load_path=load_path)
    profit.append(run)
    print("\n-----------------------------------------------------------------------------------------------\nDQN with Adam Optimzer and Experience Replay, Training Loaded")
    run = deep_q_agent(days, step, False, data, save=save, save_path=save_path, load=load, load_path=load_path)
    profit.append(run)
    print("\n-----------------------------------------------------------------------------------------------\n")
    print("Finished after: " + (str(np.round((time.time() - start_time),3))) + " seconds")


    fig, ax = plt.subplots()
    
    ax.bar(model, profit )#label=bar_labels, color=bar_colors)

    ax.set_ylabel('Profit SEK')
    ax.set_title('Profit for each method in SEK')

    plt.savefig("Comparison")
    