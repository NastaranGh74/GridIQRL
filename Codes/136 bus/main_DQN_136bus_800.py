import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from Main_RL import training_DQN
import torch




data_folder = Path("./")

LINE_DATA1 = pd.read_excel(data_folder/'Data_136bus.xlsx', sheet_name='Linedata', header=0)
BUS_DATA1 = pd.read_excel(data_folder/'Data_136bus.xlsx', sheet_name='Busdata', header=0)
P_DATA1 = pd.read_excel(data_folder/'Data_136bus.xlsx', sheet_name='P_Data', header=0, index_col=0)
Q_DATA1 = pd.read_excel(data_folder/'Data_136bus.xlsx', sheet_name='Q_Data', header=0, index_col=0)

feasible_acts1 = pd.read_csv(data_folder/'sorted_open_lines_136bus.csv', header=0, index_col=0, skiprows=3499200)
#399800
#33bus skip rows = 50671
#voltage=13.8
#tarining_steps = 31000
#epsilon_decay_value = 37e-6


stats, ep_return, moving_average, q_network, system_loss, voltage_penalty = training_DQN(P_DATA=P_DATA1, Q_DATA=Q_DATA1, LINE_DATA=LINE_DATA1, BUS_DATA=BUS_DATA1, voltage=13.8, feasible_acts=feasible_acts1, episodes=50000, alpha=0.001, batch_size=512, gamma=0.99, epsilon=1, epsilon_decay_value=22e-6)


torch.save(q_network.state_dict(), 'DQN_136_800.pt')




All_Returns = pd.DataFrame(stats['Returns'])
MSE_loss = pd.DataFrame(stats['MSE Loss'])

All_Returns.to_csv('DQN_136_800.csv')
MSE_loss.to_csv('DQN_MSE_loss_136_800.csv')

system_loss = pd.DataFrame(system_loss)
voltage_penalty = pd.DataFrame(voltage_penalty)
system_loss.to_csv('system_lossDQN_136_800.csv')
voltage_penalty.to_csv('voltage_penaltyDQN_136_800.csv')
