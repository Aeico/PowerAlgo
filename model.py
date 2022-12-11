import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims #First fully connected layer dimensions
        self.fc2_dims = fc2_dims #Second fully connected layer dimensions
        self.fc3_dims = fc3_dims #Third fully connected layer dimensions
        self.n_actions = n_actions #How many options for the Environment (House electricity choices)

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
        self.input_dims = input_dims #Amount of states from enviroment
        self.batch_size = batch_size #How much data to process in replay memory
        self.action_space = [i for i in range(n_actions)] #Amount of choices
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
        if np.random.random() > self.epsilon: #Greedy choice
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else: #Random (Exploration) choice
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

        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min   