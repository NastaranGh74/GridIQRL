from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pandapower as pp
import math


import pandapower.plotting as pplot
import seaborn
colors = seaborn.color_palette()

import copy
import gym
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from numba import jit




def training_DQN(P_DATA, Q_DATA, LINE_DATA, BUS_DATA, voltage, feasible_acts, episodes, alpha=0.0001, batch_size=128, gamma=0.99, epsilon=1, epsilon_decay_value=7e-5):
    
    ##############################################################################################
    #Function used inside powerflow
    @jit(nopython=True)
    def Count(mylist, myvalue):
        count=0
        args=[]
        for i in range(mylist.size):
            if mylist[i] == myvalue:
                count+=1
                args.append(i)
            else:
                pass
        return count, args

    ##############################################################################################
    #Powerflow
    def powerflow(LINE_DATA, BUS_DATA, line_states, P_DATA, Q_DATA, voltage, h):
        net = pp.create_empty_network()
        nbus = max(BUS_DATA.loc[:,'Bus No.'])
        nline=np.size(LINE_DATA.iloc[:,1])
    
        for i in range(nbus):
            if BUS_DATA.iloc[i,3] == 'PQ':
                pp.create_bus(net, vn_kv=voltage, name='Bus {}'.format(BUS_DATA.loc[i,'Bus No.']))
            elif BUS_DATA.iloc[i,3] == 'PV':
                pp.create_bus(net, vn_kv=voltage, name='Bus {}'.format(BUS_DATA.loc[i,'Bus No.']))
                pp.create_ext_grid(net, bus=pp.get_element_index(net, "bus", 'Bus {}'.format(BUS_DATA.loc[i,'Bus No.'])), vm_pu=1, name="Grid Connection")

        

        for i in range(nbus):
            if BUS_DATA.iloc[i,3] == 'PQ':
                pp.create_load(net, bus=pp.get_element_index(net, "bus", 'Bus {}'.format(BUS_DATA.loc[i,'Bus No.'])), 
                         p_mw=P_DATA.iloc[h,i]/1000, q_mvar=Q_DATA.iloc[h,i]/1000, name='Load {}'.format(BUS_DATA.loc[i,'Bus No.']))
      

        for i in range(nline):
            if line_states[i] == 1:
                pp.create_line_from_parameters(net, from_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA.iloc[i,1])),
                               to_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA.iloc[i,2])),
                     length_km=1,
                     r_ohm_per_km = LINE_DATA.iloc[i,3], x_ohm_per_km = LINE_DATA.iloc[i,4], c_nf_per_km = 0, max_i_ka = 99999.0, name='Line {}'.format(i), **{'type': 'ol'})
            elif line_states[i] == 0:
                pp.create_line_from_parameters(net, from_bus=pp.get_element_index(net, "bus",
                                                                  'Bus {}'.format(LINE_DATA.iloc[i,1])), to_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA.iloc[i,2])),
                     length_km=0.1,
                      r_ohm_per_km = LINE_DATA.iloc[i,3], x_ohm_per_km = LINE_DATA.iloc[i,4], c_nf_per_km = 0, max_i_ka = 99999.0, name='Line {}'.format(i), **{'type': 'ol'}, in_service = False)
      
        pp.runpp(net, algorithm='nr', calculate_voltage_angles=True, max_iteration= 500)


        ##Visualizing
        #bc = pplot.create_bus_collection(net, net.bus.index, size=80, color=colors[0], zorder=1) #create buses
        #lc = pplot.create_line_collection(net, net.line.index, color="grey", zorder=2) #create lines
        #netplot = pplot.draw_collections([lc, bc], figsize=(8,6)) # plot lines and buses


        p_tobus=[0 for _ in range(nbus)]
        p_frombus=[0 for _ in range(nbus)]
        load = P_DATA.iloc[h,:]/1000

  
  
        for j in range(nbus):
            c1, p1 = Count(LINE_DATA.iloc[:,1].values, j+1)
            c2, p2 = Count(LINE_DATA.iloc[:,2].values, j+1)
            c=c1+c2

    

            for k1 in p2:
                p_tobus[j] += net.res_line.iloc[k1,2]
  
            for k2 in p1:
                p_frombus[j] += net.res_line.iloc[k2,0]

        load_balance = np.array(p_tobus)+np.array(p_frombus)+load



        return net.res_bus.iloc[:,0], np.sum(net.res_line.iloc[:,4]),load_balance
    
    ##############################################################################################
    #Building the environment

    class Reconfig(Env):
        def __init__(self, feasible_acts, LINE_DATA, BUS_DATA, P_DATA, Q_DATA, voltage):
        
        
            #Actions are: 0:open, 1: close
            num_actions = len(feasible_acts)
            self.action_space = Discrete(num_actions)
        
        
            self.nbus = max(BUS_DATA.loc[:,'Bus No.'])
            #Line losses array (observation space)
            self.observation_space = Box(low=0, high=500,
                                             shape=((self.nbus-1)*2+len(LINE_DATA),), dtype=np.float32)
        

        
            self.dispatchable_lines=[i for i in range(1, len(LINE_DATA)+1)]
        
        
            self.action_dict_temp = dict()
        
            for i in range(1, len(self.dispatchable_lines)+1):
                self.action_dict_temp[i] = i


            self.hour = 0
            self.voltage = voltage
        
        

        def step(self, action):
        
        
            action_dict = self.action_dict_temp
        
            act = feasible_acts.iloc[action, :]
        
            self.line_states = [1] * len(LINE_DATA)
        
            for b in act:
                self.line_states[b] = 0
        

    
            self.voltages, self.loss, self.balance = powerflow(LINE_DATA, BUS_DATA, self.line_states, P_DATA, Q_DATA, self.voltage, self.hour)



            self.reward1 = 0
            self.reward2 = 0
            self.reward3 = 0
            self.reward4 = 0



            #calculate reward
            self.reward1 = -10000*self.loss



            #constraints
            #Radiality
            if sum(self.line_states) != max(BUS_DATA.loc[:,'Bus No.'])-1:
                self.reward2 = -100000

            #Demand Balance
            im_p = []
            load = P_DATA.iloc[self.hour,:]/1000
            balance_ratio = self.balance/load
            for element in balance_ratio:
                if element != float('-inf') and element != float('inf'):
                    if element > 0.2:
                        im_p.append(element)
                        self.reward3 = -100000


     
            #Voltage constraint
            for elements in self.voltages:
                if elements > 1.05:
                    self.reward4 = -1000000*(max(self.voltages)-1)
                elif elements < 0.95:
                    self.reward4 = -1000000*(1-min(self.voltages))
                elif math.isnan(elements):
                    self.reward4 = -1000000

            reward = self.reward1 + self.reward2 + self.reward3 + self.reward4
        
            #increment hour
            self.hour = self.hour + 1
            done = True
        
            
            
            #else:
             #   done = False

            
            #Set place holder for info
            info = {}
        
            self.state = [*np.around(np.array(P_DATA.iloc[self.hour,1:]), 1), *np.around(np.array(Q_DATA.iloc[self.hour,1:]), 1), *self.line_states]
            
            
            
        


            return np.array(self.state), reward, done, info

        def render(self):
            #Implement visualization
            pass

        def reset(self):

            if self.hour == 0:
                act = feasible_acts.iloc[0, :]
                self.line_states = [1] * len(LINE_DATA)
                for b in act:
                    self.line_states[b] = 0
            
            self.state = [*np.around(np.array(P_DATA.iloc[self.hour,1:]), 1), *np.around(np.array(Q_DATA.iloc[self.hour,1:]), 1), *self.line_states]

            done = False

    
            #self.voltages, self.loss, self.balance = powerflow(LINE_DATA, BUS_DATA, self.line_states, P_DATA, Q_DATA, self.hour+(episode*24))
            #Reset simulation time
            #self.sim_length = 2
    
            return np.array(self.state)

    
    ##############################################################################################
    
    env=Reconfig(feasible_acts, LINE_DATA, BUS_DATA, P_DATA, Q_DATA, voltage)
    #this should show the dimension of states that you use
    state_dims = len(env.reset())
    num_actions = env.action_space.n

    ##############################################################################################
    #Wrapper for environment
    
    class PreprocessEnv(gym.Wrapper):
    
        def __init__(self, env):
            gym.Wrapper.__init__(self, env)
        
        def reset(self):
            obs = self.env.reset()
            return torch.from_numpy(obs).unsqueeze(dim=0).float()
    
        def step(self, action):
            action = action.item()
            next_state, reward, done, info = self.env.step(action)
            next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
            reward = torch.tensor(reward).view(1, -1).float()
            done = torch.tensor(done).view(1, -1)
            return next_state, reward, done, info
    
    
    ##############################################################################################
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    env = PreprocessEnv(env)
    

    q_network = nn.Sequential(nn.Linear(state_dims, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, num_actions)).to(device)


    target_q_network = copy.deepcopy(q_network).eval().to(device)


    def policy(device, state, epsilon=0.):
        if torch.rand(1) < epsilon:
            return torch.randint(num_actions, (1, 1))
        else:
            av = q_network(state.to(device)).detach()
            return torch.argmax(av, dim=-1, keepdim=True)
    
    
    ##############################################################################################
    #Memory to store transitions
    
    class ReplayMemory:
    
        def __init__(self, capacity=30000000):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def insert(self, transition):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            assert self.can_sample(batch_size)

            batch = random.sample(self.memory, batch_size)
            batch = zip(*batch)
            return [torch.cat(items) for items in batch]

        def can_sample(self, batch_size):
            return len(self.memory) >= batch_size * 10

        def __len__(self):
            return len(self.memory)
    
    
    
    ##############################################################################################################
    #Main learning loop
    
    
    def deep_q_learning(device, q_network, policy, episodes, alpha=0.0001, batch_size=128, gamma=0.99, epsilon=1, epsilon_decay_value=7e-5):
        optim = AdamW(q_network.parameters(), lr=alpha)
        memory = ReplayMemory()
        stats = {'MSE Loss':[], 'Returns':[]}
        
        
        
    
        END_EPSILON_DECAYING = 0.00001
        START_EPSILON_DECAYING = 1
        #epsilon_decay_value = 7e-5
        system_loss = []
        voltage_penalty = []
    
        for episode in tqdm(range(1, episodes+1)):
            state = env.reset()
            done = False
            ep_return = 0
        
        
            #Decaying epsilon
            if END_EPSILON_DECAYING <= epsilon <= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
        
            while not done:
                action = policy(device, state.to(device), epsilon)
            
                next_state, reward, done, _ = env.step(action.cpu())
                memory.insert([state.cpu(), action.cpu(), reward.cpu(), done.cpu(), next_state.cpu()])
                system_loss.append(env.loss)
                voltage_penalty.append(env.reward4)
            
                if memory.can_sample(batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                    qsa_b = q_network(state_b.to(device)).gather(1, action_b.to(device))
                
                    #next_qsa_b = target_q_network(next_state_b)
                    #next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0] #pytorch when sees a single element in a dimension tries to eliminate that dimension so we use keepdim=True
                
                    target_b = reward_b 
                    #+ ~done*gamma*next_qsa_b
                    loss = F.mse_loss(qsa_b.to(device), target_b.to(device))
                
                    q_network.zero_grad()
                    loss.backward()
                    optim.step()
                
                    stats['MSE Loss'].append(loss.item())
                
                state = next_state.cpu()
                ep_return += reward.item()
              
            stats['Returns'].append(ep_return)
        
            if episode%10 == 0:
                target_q_network.load_state_dict(q_network.state_dict())
            
        return stats, system_loss, voltage_penalty




    stats, system_loss, voltage_penalty = deep_q_learning(device, q_network, policy, episodes=episodes)
    
    ####################################################################################################################
    #Generating necessary plots
    
    ep_return = []
    for i in range(int(len(stats['Returns'])/24)):
        ep_return.append(np.mean(stats['Returns'][i*24:i*24 + 23]))
    
    plt.figure(1)
    plt.plot(ep_return)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    
    
    moving_average = []
    for i in range(len(ep_return)-30):
        moving_average.append(np.mean(ep_return[i:i+30]))
    
    plt.figure(2)
    plt.plot(moving_average)
    plt.xlabel('Episode')
    plt.ylabel('Reward 30 Step Moving Average')
    
    
    return stats, ep_return, moving_average, q_network, system_loss, voltage_penalty
    