import numpy as np
from dueling_dqn import Agent
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from environment import Reconfig





def main():
    
    data_folder = Path("./")

    LINE_DATA1 = pd.read_excel(data_folder/'Data_33bus.xlsx', sheet_name='Linedata', header=0)
    BUS_DATA1 = pd.read_excel(data_folder/'Data_33bus.xlsx', sheet_name='Busdata', header=0)
    P_DATA1 = pd.read_excel(data_folder/'Data_33bus.xlsx', sheet_name='P_Data', header=0, index_col=0)
    Q_DATA1 = pd.read_excel(data_folder/'Data_33bus.xlsx', sheet_name='Q_Data', header=0, index_col=0)
    FEASible_ACTS = pd.read_csv(data_folder/'sorted_open_lines_33bus.csv', header=0, index_col=0, skiprows=50451)
    
    NBUS = len(BUS_DATA1)
    N_ACTS = len(FEASible_ACTS)

    VOLTAGE = 12.66
    NUM_GAMES = 30000
    
    
    env = Reconfig(feasible_acts=FEASible_ACTS, LINE_DATA=LINE_DATA1, BUS_DATA=BUS_DATA1, P_DATA=P_DATA1 , Q_DATA=Q_DATA1, voltage=VOLTAGE)
    state_dims = len(env.reset())
    num_actions = env.action_space.n
    

    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.001,
                  state_shape=[state_dims], fc1_dim=1024, fc2_dim=1024, n_actions=num_actions, capacity=3000000, eps_min=0.001,
                  batch_size=512, eps_dec=37e-6, replace=10)
    


    all_rewards = []
    system_loss = []
    voltage_penalty = []
    
    ############################################################################################################################
    #Training Loop
    for i in tqdm(range(NUM_GAMES)):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            system_loss.append(env.loss)
            voltage_penalty.append(env.reward4)
            
            all_rewards.append(reward)
            agent.store_transition(observation, action,
                                    reward, observation_, int(done))
            agent.learn()

            observation = observation_


    save_results(all_rewards, agent, system_loss, voltage_penalty, NBUS, N_ACTS)
    
    
def save_results(rewards, agent_, system_loss, voltage_penalty, nbus, n_acts):
    
    rewards = pd.DataFrame(rewards)
    rewards.to_csv(f"DuelingDQN_{nbus}_{n_acts}.csv")
    
    torch.save(agent_.q_eval.state_dict(), f"DuelingDQN_{nbus}_{n_acts}.pt")
    
    system_loss = pd.DataFrame(system_loss)
    system_loss.to_csv(f"system_lossDDQN_{nbus}_{n_acts}.csv")
    
    voltage_penalty = pd.DataFrame(voltage_penalty)
    voltage_penalty.to_csv(f"voltage_penaltyDDQN_{nbus}_{n_acts}.csv")

    
    
    
if __name__ == '__main__':
    main()