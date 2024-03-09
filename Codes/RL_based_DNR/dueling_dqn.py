import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ExperienceReplayBuffer(object):
    def __init__(self, capacity, state_shape, n_actions):
        self.capacity = capacity
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.new_state_mem = np.zeros((self.capacity, *state_shape), dtype=np.float32)
        self.action_mem = np.zeros(self.capacity, dtype=np.int64)
        self.reward_mem = np.zeros(self.capacity, dtype=np.float32)
        self.terminal_mem = np.zeros(self.capacity, dtype=np.uint8)

    def store_trajectory(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.capacity
        self.state_mem[index] = state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.new_state_mem[index] = state_
        self.terminal_mem[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.capacity)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        terminal = self.terminal_mem[batch]

        return states, actions, rewards, states_, terminal


class DuelingLinearDeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_actions, name, state_shape, fc1_dim, fc2_dim):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(*state_shape, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        if T.cuda.is_available():
            device = T.device('cuda')
        else:
            device = T.device('cpu')
            
        self.device = device
        self.to(self.device)
        

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A



class Agent(object):
    def __init__(self, gamma, epsilon, alpha, n_actions, state_shape, fc1_dim, fc2_dim,
                 capacity, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.memory = ExperienceReplayBuffer(capacity, state_shape, n_actions)

        self.q_eval = DuelingLinearDeepQNetwork(alpha, n_actions, state_shape=state_shape,
                                   name='q_eval', fc1_dim=fc1_dim, fc2_dim=fc2_dim)
        self.q_next = DuelingLinearDeepQNetwork(alpha, n_actions, state_shape=state_shape,
                                   name='q_next', fc1_dim=fc1_dim, fc2_dim=fc2_dim)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_trajectory(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis,:]
            state = T.tensor(observation, dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
                         

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
                                self.memory.sample_buffer(self.batch_size)

        
        state = T.tensor(state).to(self.q_eval.device)
        new_state = T.tensor(new_state).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        V_s, A_s = self.q_eval.forward(state)
        V_s_, A_s_ = self.q_next.forward(new_state)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1,
                                              action.unsqueeze(-1)).squeeze(-1)

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*T.max(q_next, dim=1)[0].detach()
        

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
        
        return
