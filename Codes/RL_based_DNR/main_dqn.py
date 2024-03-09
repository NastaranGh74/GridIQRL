import numpy as np
import pandas as pd
from pathlib import Path
from dqn import PreprocessEnv, ReplayMemory
from environment import Reconfig
from tqdm import tqdm
import torch
import copy




def policy(device, num_actions, q_network, state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    else:
        av = q_network(state.to(device)).detach()
        return torch.argmax(av, dim=-1, keepdim=True)

##############################################################################################################
#Training Loop  

def deep_q_learning(env, device, num_actions, q_network, policy, episodes, alpha=0.0001, batch_size=128, gamma=0.99, epsilon=1, epsilon_decay_value=7e-5):
    
    target_q_network = copy.deepcopy(q_network).eval().to(device)
    optim = torch.optim.AdamW(q_network.parameters(), lr=alpha)
    memory = ReplayMemory()

    END_EPSILON_DECAYING = 0.00001
    START_EPSILON_DECAYING = 1
    system_loss = []
    voltage_penalty = []
    NN_loss = []
    all_rewards = []

    for episode in tqdm(range(1, episodes+1)):
        state = env.reset()
        done = False

        #Decaying epsilon
        if END_EPSILON_DECAYING <= epsilon <= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

        while not done:
            action = policy(device, num_actions, q_network, state.to(device), epsilon)

            next_state, reward, done, _ = env.step(action.cpu())
            all_rewards.append(reward.item())
            memory.insert([state.cpu(), action.cpu(), reward.cpu(), done.cpu(), next_state.cpu()])
            system_loss.append(env.loss)
            voltage_penalty.append(env.reward4)

            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                qsa_b = q_network(state_b.to(device)).gather(1, action_b.to(device))
                target_b = reward_b 
                loss = torch.nn.functional.mse_loss(qsa_b.to(device), target_b.to(device))
                q_network.zero_grad()
                loss.backward()
                optim.step()

                NN_loss.append(loss.item())

            state = next_state.cpu()

        if episode%10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

    return all_rewards, NN_loss, system_loss, voltage_penalty


####################################################################################################################
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
    data_folder = Path("./")
    
    #Define NN layer neurons
    fc1_dim = 1024
    fc2_dim =1024


    ##############################################################################################
    #Define the environment
    env = Reconfig(feasible_acts=FEASible_ACTS, LINE_DATA=LINE_DATA1, BUS_DATA=BUS_DATA1, P_DATA=P_DATA1 , Q_DATA=Q_DATA1, voltage=VOLTAGE)
    state_dims = len(env.reset())
    num_actions = env.action_space.n
    ##############################################################################################
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    env = PreprocessEnv(env)
    
    ###################################################################################################
    #Define NN network
    q_network = torch.nn.Sequential(torch.nn.Linear(state_dims, fc1_dim),
                           torch.nn.ReLU(),
                           torch.nn.Linear(fc1_dim, fc2_dim),
                           torch.nn.ReLU(),
                           torch.nn.Linear(fc2_dim, num_actions)).to(device)

    
    
    all_rewards, NN_loss, system_loss, voltage_penalty = deep_q_learning(env, device, num_actions, q_network, policy, episodes=NUM_GAMES)

    save_results(all_rewards, q_network, NN_loss, system_loss, voltage_penalty, NBUS, N_ACTS)

    
#########################################################################################################
def save_results(rewards, agent_, NN_loss, system_loss, voltage_penalty, nbus, n_acts):
    
    rewards = pd.DataFrame(rewards)
    rewards.to_csv(f"DQN_{nbus}_{n_acts}.csv")
    
    torch.save(agent_.state_dict(), f"DQN_{nbus}_{n_acts}.pt")
    
    NN_loss = pd.DataFrame(NN_loss)
    NN_loss.to_csv(f"DQN_MSE_loss_{nbus}_{n_acts}.csv")
    
    system_loss = pd.DataFrame(system_loss)
    system_loss.to_csv(f"system_lossDQN_{nbus}_{n_acts}.csv")
    
    voltage_penalty = pd.DataFrame(voltage_penalty)
    voltage_penalty.to_csv(f"voltage_penaltyDQN_{nbus}_{n_acts}.csv")
    
    
    
if __name__ == '__main__':
    main()