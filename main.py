# %%
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import homeenv

Home_Enviroment = homeenv.Home_Enviroment


import json
import time


import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def plot_env(x, charge, exchange, filename, should_show, lines=None):
    fig=plt.figure()

    x1=fig.add_subplot(111, label="1")
    x2=fig.add_subplot(111, label="2", frame_on=False)
    
    x1.plot(x, exchange, color="C0")
    x1.set_xlabel("Days", color="k")
    x1.set_ylabel("Exchange", color="C0")
    x1.tick_params(axis='x', colors="k")
    x1.tick_params(axis='y', colors="C0")

    x2.plot(x, charge, color="C1")
    x2.set_ylabel("Charge", color="C1")
    x2.yaxis.set_label_position('right')
    x2.yaxis.tick_right()
    x2.tick_params(axis='x', colors="k")
    x2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    if(should_show == True):
        plt.show()

def plot_env_score(x, score, filename, should_show, lines=None):
    
    fig=plt.figure()

    x1=fig.add_subplot(111, label="1")
    x2=fig.add_subplot(111, label="2", frame_on=False)
    #, path_effects=[pe.withStroke(linewidth=3, foreground="black")]
    x1.plot(x, score, color="C2")
    x1.set_xlabel("Days", color="k")
    x1.set_ylabel("Score", color="C2")
    x1.tick_params(axis='x', colors="C2")
    x1.tick_params(axis='y', colors="C2")

    arr_mean_price = []
    temp = []
    for i in range(len(x)):
        for j in range(24):
            temp.append(data[j*(i+1)]['Value'])
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
    #Home Enviroment

    env = Home_Enviroment(step_period,data)
    counts = [0,0,0]
    state = env.init()
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

def only_sell(days, step_period, show, data):
    env = Home_Enviroment(step_period,data)
    counts = [0,0,0]
    while env.day_index <= days:#((len(data)/24)):
        state, reward, done = env.step(0)
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i+1 for i in range(days+1)]
    filename = 'onlystepsell.png'

    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)

def only_buy(days, step_period, show, data):
    env = Home_Enviroment(step_period,data)
    counts = [0,0,0]
    days = days
    while env.day_index <= days:#((len(data)/24)):
        state, reward, done = env.step(1)
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i+1 for i in range(days+1)]
    filename = 'onlystepbuy.png'
    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)

def test_rewards(days, step_period, show, data):
    env = Home_Enviroment(step_period,data)
    counts = [0,0,0]
    days = days
    while env.day_index <= days:#((len(data)/24)):
        first_diff = (env.data[0+int((env.time/3600))]['Value']) - (env.data[1+int((env.time/3600))]['Value']) 

        second_diff = (env.data[1+int((env.time/3600))]['Value']) - (env.data[2+int((env.time/3600))]['Value'])

        if first_diff + second_diff > -10 or env.home_charge + env.home_discharge_rate_temp*env.step_period + env.solar_charge_rate_temp*env.step_period >= env.max_charge:
            state, reward, done = env.step(0)
        else: 
            state, reward, done = env.step(1)
    ampere_hour = env.home_charge/3600
    print(f"{np.round(env.total_sold_price/100,3)}SEK Sold and {np.round(env.total_bought_price/100,3)}SEK Bought  |  Delta = {np.round(env.total_sold_price/100 - env.total_bought_price/100,3)}kr | Charge at {np.round(ampere_hour,3)}Ah")
    x = [i+1 for i in range(days+1)]
    filename = 'reward_test.png'
    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)

    
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims #First fully connected layer
        self.fc2_dims = fc2_dims #Second fully connected layer
        self.fc3_dims = fc3_dims #Third fully connected layer
        self.n_actions = n_actions #How many options for the enviroment (House electricity choices)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) #Linear layers
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr) 
        self.loss = nn.MSELoss() #Loss function, MSELoss is mean squared error
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #Device agonstic for gpu and non gpu
        self.to(self.device) #Set to cuda if cuda and cpu if cpu

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, 
                batch_size, n_actions, eps_end, eps_dec,
                load, load_path, max_mem_size=200000):#Large memory size since 5000+ days is 120000+ hours and we want to have some memory
                #Even though we cant remember all minutes or seconds
        self.gamma = gamma #discounts reward
        self.epsilon = epsilon #increased learning rate early on
        self.eps_min = eps_end #ending numer of increased learning rate
        self.eps_dec = eps_dec #decrement of increased learning rate
        self.lr = lr #learning rate
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=64, fc2_dims=64, fc3_dims=32)

        #Loading model
        if load:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.Q_eval.load_state_dict(torch.load(load_path,device))

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.day_done_memory = np.zeros(self.mem_size, dtype=bool)
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.day_done_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else: 
            action = np.random.choice(self.action_space)
        
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        day_done_batch = torch.tensor(self.day_done_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[day_done_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] #Get max value in dimension one, keep that value
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                                                    else self.eps_min                                

def deep_q_agent(days, step_period, show, data, save, save_path, load, load_path):
    env = Home_Enviroment(step_period,data)
    agent = Agent(gamma=0.99, epsilon=0.01, batch_size=32, n_actions=2, 
                eps_end=0.001, eps_dec=5e-6, input_dims=[4], lr=0.001,
                load_path=load_path, load=load)
    scores, eps_history = [],[]
    action_count = [0,0,0]

    for i in range(days):
        score = 0
        done = False
        observation = env.init()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action)
            score += reward
            #print(reward)
            action_count[action] += 1
            #print(str(observation[0]) + ' ' + str(action))
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
        mean_scores.append(np.mean(scores[-5:i]))

    plot_env(x,env.charge_at_time,(env.exchange),filename,should_show=show)
    plot_env_score(x, scores, 'agent_score.png',should_show=show)

    print(action_count)
    
    if save:
        torch.save(agent.Q_eval.state_dict(), save_path)

#All functions use (Days, Step_Period(60 = 1 min), Show graph)
if __name__ == "__main__":
    start_time = time.time()

    #file = open('2008till2022-10-15.json')
    file = open('2021till2022nov.json')

    data = json.load(file)
    
    save = False
    load = True
    save_path = "Q_Eval_Days4k_Info3h_Step10min2.pth"
    load_path = "Q_Eval_Days4k_Info3h_Step10min2.pth"

    days = 670
    step = 60*60
    print("\n-----------------------------------------------------------------------------------------------\n")
    if_step_sell(days, step, False, data)
    print("\n-----------------------------------------------------------------------------------------------\n")
    only_sell(days, step, False, data)
    print("\n-----------------------------------------------------------------------------------------------\n")
    only_buy(days, step, False, data)
    print("\n-----------------------------------------------------------------------------------------------\n")
    test_rewards(days, step, False, data)
    print("\n-----------------------------------------------------------------------------------------------\n")
    deep_q_agent(days, step, False, data, save=save, save_path=save_path, load=load, load_path=load_path)
    print("\n-----------------------------------------------------------------------------------------------\n")
    print("Finished after: " + (str(np.round((time.time() - start_time),3))) + " seconds")
    