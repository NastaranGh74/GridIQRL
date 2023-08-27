import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import copy
import math


import gym
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse


from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pandapower as pp
from torch import multiprocessing as mp

import pandapower.networks as nw
import pandapower.plotting as pplot
import seaborn
colors = seaborn.color_palette()

from tqdm import tqdm

from torch.autograd import Variable





data_folder = Path("./")
P_DATA = pd.read_excel(data_folder/'Data.xlsx', sheet_name='P_Data', header=0, index_col=0)
Q_DATA = pd.read_excel(data_folder/'Data.xlsx', sheet_name='Q_Data', header=0, index_col=0)

LINE_DATA = pd.read_excel(data_folder/'Data.xlsx', sheet_name='Linedata', header=0)
BUS_DATA = pd.read_excel(data_folder/'Data.xlsx', sheet_name='Busdata', header=0)




allP=pd.DataFrame()
allQ=pd.DataFrame()

for i in tqdm(range(33)):
    bus_i_P = pd.DataFrame()
    bus_i_Q = pd.DataFrame()
    for j in tqdm(range(3001)):
        data2P=pd.DataFrame(np.random.normal(loc=P_DATA.iloc[:, i], scale=0.15*P_DATA.iloc[:, i]))
        bus_i_P = pd.concat([bus_i_P, data2P], axis=0)
        
        data2Q=pd.DataFrame(np.random.normal(loc=Q_DATA.iloc[:, i], scale=0.15*Q_DATA.iloc[:, i]))
        bus_i_Q = pd.concat([bus_i_Q, data2Q], axis=0)
        
    allP = pd.concat([allP, bus_i_P], axis=1)
    allQ = pd.concat([allQ, bus_i_Q], axis=1)
    
    
allP.columns = P_DATA.columns.values
allQ.columns = Q_DATA.columns.values



writer = pd.ExcelWriter('Data2_33bus.xlsx', engine='xlsxwriter')
workbook = writer.book

allP.to_excel(writer,
             sheet_name="P_Data",
             startrow=0,
             startcol=0)

allQ.to_excel(writer,
             sheet_name="Q_Data",
             startrow=0,
             startcol=0)


LINE_DATA.to_excel(writer,
             sheet_name="Linedata",
             startrow=0,
             startcol=0)

BUS_DATA.to_excel(writer,
             sheet_name="Busdata",
             startrow=0,
             startcol=0)
writer.save()