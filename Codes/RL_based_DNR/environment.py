import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import pandas as pd
import pandapower as pp
import math

class Reconfig(Env):
    def __init__(self, feasible_acts, LINE_DATA, BUS_DATA, P_DATA, Q_DATA, voltage):

        self.LINE_DATA = LINE_DATA
        self.BUS_DATA = BUS_DATA
        self.P_DATA = P_DATA
        self.Q_DATA = Q_DATA
        self.feasible_acts = feasible_acts
        self.voltage = voltage


        #Actions are: 0:open, 1: close
        num_actions = len(self.feasible_acts)
        self.action_space = Discrete(num_actions)


        self.nbus = max(self.BUS_DATA.loc[:,'Bus No.'])
        
        self.observation_space = Box(low=0, high=500,
                                        shape=((self.nbus-1)*2+len(self.LINE_DATA),), dtype=np.float32)



        self.dispatchable_lines=[i for i in range(1, len(self.LINE_DATA)+1)]


        self.action_dict_temp = dict()

        for i in range(1, len(self.dispatchable_lines)+1):
            self.action_dict_temp[i] = i


        self.hour = 0



    ##############################################################################################
    ## Function to count occurrences of a value in a list

    def Count(self, mylist, myvalue):
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
    def powerflow(self, LINE_DATA2, BUS_DATA2, line_states2, P_DATA2, Q_DATA2, voltage2, h):

        net = pp.create_empty_network()
        nbus = max(BUS_DATA2.loc[:,'Bus No.'])
        nline=np.size(LINE_DATA2.iloc[:,1])

        for i in range(nbus):
            if BUS_DATA2.iloc[i,3] == 'PQ':
                pp.create_bus(net, vn_kv=voltage2, name='Bus {}'.format(BUS_DATA2.loc[i,'Bus No.']))
            elif BUS_DATA2.iloc[i,3] == 'PV':
                pp.create_bus(net, vn_kv=voltage2, name='Bus {}'.format(BUS_DATA2.loc[i,'Bus No.']))
                pp.create_ext_grid(net, bus=pp.get_element_index(net, "bus", 'Bus {}'.format(BUS_DATA2.loc[i,'Bus No.'])), vm_pu=1, name="Grid Connection")



        for i in range(nbus):
            if BUS_DATA2.iloc[i,3] == 'PQ':
                pp.create_load(net, bus=pp.get_element_index(net, "bus", 'Bus {}'.format(BUS_DATA2.loc[i,'Bus No.'])), 
                        p_mw=P_DATA2.iloc[h,i]/1000, q_mvar=Q_DATA2.iloc[h,i]/1000, name='Load {}'.format(BUS_DATA2.loc[i,'Bus No.']))


        for i in range(nline):
            if line_states2[i] == 1:
                pp.create_line_from_parameters(net, from_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA2.iloc[i,1])),
                                               to_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA2.iloc[i,2])),
                        length_km=1,
                    r_ohm_per_km = LINE_DATA2.iloc[i,3], x_ohm_per_km = LINE_DATA2.iloc[i,4], c_nf_per_km = 0, max_i_ka = 99999.0, name='Line {}'.format(i), **{'type': 'ol'})
            elif line_states2[i] == 0:
                pp.create_line_from_parameters(net, from_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA2.iloc[i,1])),
                                               to_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA2.iloc[i,2])),
                        length_km=1,
                    r_ohm_per_km = LINE_DATA2.iloc[i,3], x_ohm_per_km = LINE_DATA2.iloc[i,4], c_nf_per_km = 0, max_i_ka = 99999.0, name='Line {}'.format(i), **{'type': 'ol'},
                                               in_service = False)

        pp.runpp(net, algorithm='nr', calculate_voltage_angles=True, max_iteration= 500)


        ##Visualizing
        #bc = pplot.create_bus_collection(net, net.bus.index, size=80, color=colors[0], zorder=1) #create buses
        #lc = pplot.create_line_collection(net, net.line.index, color="grey", zorder=2) #create lines
        #netplot = pplot.draw_collections([lc, bc], figsize=(8,6)) # plot lines and buses


        p_tobus=[0 for _ in range(nbus)]
        p_frombus=[0 for _ in range(nbus)]
        load = P_DATA2.iloc[h,:]/1000



        for j in range(nbus):
            c1, p1 = self.Count(LINE_DATA2.iloc[:,1].values, j+1)
            c2, p2 = self.Count(LINE_DATA2.iloc[:,2].values, j+1)
            c=c1+c2



            for k1 in p2:
                p_tobus[j] += net.res_line.iloc[k1,2]

            for k2 in p1:
                p_frombus[j] += net.res_line.iloc[k2,0]

        load_balance = np.array(p_tobus)+np.array(p_frombus)+load



        return net.res_bus.iloc[:,0], np.sum(net.res_line.iloc[:,4]),load_balance

    ##############################################################################################
    #Building the environment


    def step(self, action):


        action_dict = self.action_dict_temp

        act = self.feasible_acts.iloc[action, :]

        self.line_states = [1] * len(self.LINE_DATA)

        for b in act:
            self.line_states[b] = 0



        self.voltages, self.loss, self.balance = self.powerflow(self.LINE_DATA, self.BUS_DATA, self.line_states, self.P_DATA, self.Q_DATA, self.voltage, self.hour)




        self.reward1 = 0
        self.reward2 = 0
        self.reward3 = 0
        self.reward4 = 0



        #calculate reward
        self.reward1 = -10000*self.loss



        #constraints
        #Radiality
        if sum(self.line_states) != max(self.BUS_DATA.loc[:,'Bus No.'])-1:
            self.reward2 = -100000

        #Demand Balance
        im_p = []
        load = self.P_DATA.iloc[self.hour,:]/1000
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


        #Set place holder for info
        info = {}

        self.state = [*np.around(np.array(self.P_DATA.iloc[self.hour,1:]), 1), *np.around(np.array(self.Q_DATA.iloc[self.hour,1:]), 1), *self.line_states]


        return np.array(self.state), reward, done, info
    

    def render(self):
        #Implement visualization
        pass
    

    def reset(self):

        if self.hour == 0:
            act = self.feasible_acts.iloc[0, :]
            self.line_states = [1] * len(self.LINE_DATA)
            for b in act:
                self.line_states[b] = 0


        self.state = [*np.around(np.array(self.P_DATA.iloc[self.hour,1:]), 1), *np.around(np.array(self.Q_DATA.iloc[self.hour,1:]), 1), *self.line_states]

        done = False


        return np.array(self.state)
