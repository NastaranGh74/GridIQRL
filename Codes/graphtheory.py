import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
from pathlib import Path
import networkx as nx
import yamada
from numba import jit


LINE_DATA = pd.read_excel('Data_33bus.xlsx', sheet_name='Linedata', header=0)
BUS_DATA = pd.read_excel('Data_33bus.xlsx', sheet_name='Busdata', header=0)




##############################################################################################
#Function used inside powerflow
def Count(mylist, myvalue):
    count=0
    args=[]
    for i in range(np.size(mylist)):
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
            pp.create_line_from_parameters(net, from_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA.iloc[i,1])),
                           to_bus=pp.get_element_index(net, "bus", 'Bus {}'.format(LINE_DATA.iloc[i,2])),
                    length_km=1,
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



    return net.res_bus.iloc[:,0], np.sum(net.res_line.iloc[:,4]),load_balance, net



line_states  = [1] * len(LINE_DATA)
voltages, loss, balance, net = powerflow(LINE_DATA, BUS_DATA, line_states, pd.DataFrame(BUS_DATA.loc[:,'P_load']).transpose(), pd.DataFrame(BUS_DATA.loc[:,'Q_load']).transpose(), voltage=12.66, h=0)


G = nx.Graph()
initial_line_states = [1] * len(LINE_DATA)
    
#The graph is fully connected at first
for i in range(len(LINE_DATA)):
    if initial_line_states[i] == 1:
        G.add_edges_from([(LINE_DATA['From bus'][i], LINE_DATA['To bus'][i])], weight=1)
        
        
#act = [135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155] #base case
#act = [117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131]
act = [32,33,34,35,36]
initial_tree = nx.Graph()
for i in act:
    initial_line_states[i]=0
    
for i in range(len(LINE_DATA)):
    if initial_line_states[i] == 1:
        initial_tree.add_edges_from([(LINE_DATA['From bus'][i], LINE_DATA['To bus'][i])], weight=1)
 
    
    

# retrieve all minimum spanning trees 
graph_yamada = yamada.Yamada(G, n_trees=3500000)
all_msts = graph_yamada.spanning_trees(initial_tree)


 
#shutil.rmtree('DataBase/')


@jit(nopython=True)
def cal(all_msts,myarray,lossval):
    all_loss = []
    all_lines = []
    all_open_lines = []
    for j in range(len(all_msts)):
        edgedata = all_msts[j]
        lines = []
        open_lines = []
        loss = 0
        for i in range(len(edgedata)):
            #line_no = LINE_DATA[((LINE_DATA['From bus']==edgedata[i][0]) & (LINE_DATA['To bus']==edgedata[i][1])) | ((LINE_DATA['From bus']==edgedata[i][1]) & (LINE_DATA['To bus']==edgedata[i][0]))]['Line Number'].values[0]-1
            line_no = int(myarray[((myarray[:,1]==edgedata[i][0]) & (myarray[:,2]==edgedata[i][1])) | ((myarray[:,1]==edgedata[i][1]) & (myarray[:,2]==edgedata[i][0]))][0][0]-1)
            loss += lossval[line_no,4]
            lines.append(line_no)
        all_loss.append(loss)
        all_lines.append(lines)
        for k in range(len(myarray)):
            if k not in lines:
                open_lines.append(k)
        all_open_lines.append(open_lines)
        
    return all_open_lines, all_lines, all_loss

lossval = net.res_line.to_numpy()
myarray = LINE_DATA.to_numpy()
all_msts = np.array(all_msts)
all_open_lines, all_lines, all_loss = cal(all_msts,myarray,lossval)
    
sorted_lines = [x for _,x in sorted(zip(all_loss,all_lines))]
sorted_open_lines = [x for _,x in sorted(zip(all_loss,all_open_lines))]
sorted_loss = np.sort(all_loss)
    
sorted_open_lines = pd.DataFrame(sorted_open_lines)
sorted_open_lines.to_csv('sorted_open_lines_33busnew.csv')

sorted_lines = pd.DataFrame(sorted_lines)
sorted_lines.to_csv('sorted_closed_lines_33busnew.csv')

sorted_loss = pd.DataFrame(sorted_loss)
sorted_loss.to_csv('sorted_loss_33busnew.csv')
