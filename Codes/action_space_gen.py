import numpy as np
import pandas as pd
import pandapower as pp
import networkx as nx
from pathlib import Path
import yamada
from numba import jit



# Function to count occurrences of a value in a list
def count_occurrences(mylist, myvalue):
    count = 0
    indices = []
    for i, val in enumerate(mylist):
        if val == myvalue:
            count += 1
            indices.append(i)
    return count, indices


# Power system modeling and Powerflow
def powerflow(LINE_DATA, BUS_DATA, line_states, P_DATA, Q_DATA, voltage, h):
    net = pp.create_empty_network()
    nbus = max(BUS_DATA.loc[:, 'Bus No.'])
    nline = np.size(LINE_DATA.iloc[:, 1])

    for i in range(nbus):
        if BUS_DATA.iloc[i, 3] in ['PQ', 'PV']:
            pp.create_bus(net, vn_kv=voltage, name=f'Bus {BUS_DATA.loc[i, "Bus No."]}')
            if BUS_DATA.iloc[i, 3] == 'PV':
                pp.create_ext_grid(net, bus=pp.get_element_index(net, "bus", f'Bus {BUS_DATA.loc[i, "Bus No."]}'),
                                   vm_pu=1, name="Grid Connection")

    for i in range(nbus):
        if BUS_DATA.iloc[i, 3] == 'PQ':
            pp.create_load(net, bus=pp.get_element_index(net, "bus", f'Bus {BUS_DATA.loc[i, "Bus No."]}'),
                           p_mw=P_DATA.iloc[h, i] / 1000, q_mvar=Q_DATA.iloc[h, i] / 1000,
                           name=f'Load {BUS_DATA.loc[i, "Bus No."]}')

    for i in range(nline):
        line_type = 'ol' if line_states[i] == 1 else 'ol', {'in_service': False}
        pp.create_line_from_parameters(net, from_bus=pp.get_element_index(net, "bus", f'Bus {LINE_DATA.iloc[i, 1]}'),
                                       to_bus=pp.get_element_index(net, "bus", f'Bus {LINE_DATA.iloc[i, 2]}'),
                                       length_km=1, r_ohm_per_km=LINE_DATA.iloc[i, 3], x_ohm_per_km=LINE_DATA.iloc[i, 4],
                                       c_nf_per_km=0, max_i_ka=99999.0, name=f'Line {i}', type=line_type)

    pp.runpp(net, algorithm='nr', calculate_voltage_angles=True, max_iteration=500)

    # Calculate load balance
    load = P_DATA.iloc[h, :].values / 1000
    p_tobus = np.zeros(nbus)
    p_frombus = np.zeros(nbus)

    for j in range(nbus):
        c1, p1 = count_occurrences(LINE_DATA.iloc[:, 1].values, j + 1)
        c2, p2 = count_occurrences(LINE_DATA.iloc[:, 2].values, j + 1)
        c = c1 + c2

        for k1 in p2:
            p_tobus[j] += net.res_line.iloc[k1, 2]

        for k2 in p1:
            p_frombus[j] += net.res_line.iloc[k2, 0]

    load_balance = p_tobus + p_frombus + load

    return net.res_bus.iloc[:, 0], np.sum(net.res_line.iloc[:, 4]), load_balance, net


# Create initial graph
def create_initial_graph(LINE_DATA, initial_line_states):
    G = nx.Graph()
    for i, state in enumerate(initial_line_states):
        if state == 1:
            G.add_edges_from([(LINE_DATA['From bus'][i], LINE_DATA['To bus'][i])], weight=1)
    return G




# Function to calculate all open lines, closed lines, and their respective losses
@jit(nopython=True)
def calculate_losses(all_msts, myarray, lossval):
    all_loss = []
    all_lines = []
    all_open_lines = []
    for j in range(len(all_msts)):
        edgedata = all_msts[j]
        lines = []
        open_lines = []
        loss = 0
        for i in range(len(edgedata)):
            line_no = int(myarray[((myarray[:, 1] == edgedata[i][0]) & (myarray[:, 2] == edgedata[i][1])) |
                                   ((myarray[:, 1] == edgedata[i][1]) & (myarray[:, 2] == edgedata[i][0]))][0][0] - 1)
            loss += lossval[line_no, 4]
            lines.append(line_no)
        all_loss.append(loss)
        all_lines.append(lines)
        for k in range(len(myarray)):
            if k not in lines:
                open_lines.append(k)
        all_open_lines.append(open_lines)

    return all_open_lines, all_lines, all_loss



def save_results(sorted_open_lines, sorted_lines, sorted_loss, nbus):
    sorted_open_lines = pd.DataFrame(sorted_open_lines)
    sorted_open_lines.to_csv(f"sorted_open_lines_{nbus}bus.csv")

    sorted_lines = pd.DataFrame(sorted_lines)
    sorted_lines.to_csv(f"sorted_closed_lines_{nbus}bus.csv")

    sorted_loss = pd.DataFrame(sorted_loss)
    sorted_loss.to_csv(f"sorted_loss_{nbus}bus.csv")



def main():
    data_folder = Path("./")

    # Read data
    LINE_DATA = pd.read_excel(data_folder / 'Data_33bus.xlsx', sheet_name='Linedata', header=0)
    BUS_DATA = pd.read_excel(data_folder / 'Data_33bus.xlsx', sheet_name='Busdata', header=0)
    #Define number of trees to find
    N_TREES = 10#3500000
    VOLTAGE = 12.66
    Num_Bus = len(BUS_DATA)
    
    ACT = [32, 33, 34, 35, 36]  # Base tree case for 33-bus system, numbers represent open lines
    #ACT = [135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155] #base case for 136-bus system
    #ACT = [117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131] #base case for 119-bus system


    #The graph is fully connected at first
    line_states = [1] * len(LINE_DATA)
    voltages, loss, balance, net = powerflow(LINE_DATA, BUS_DATA, line_states, pd.DataFrame(BUS_DATA.loc[:, 'P_load']).transpose(),
                                         pd.DataFrame(BUS_DATA.loc[:, 'Q_load']).transpose(), voltage=VOLTAGE, h=0)

    # Graph creation and MST generation
    G = create_initial_graph(LINE_DATA, [1] * len(LINE_DATA))
    
    initial_line_states = [1] * len(LINE_DATA)
    for i in ACT:
        initial_line_states[i] = 0
    initial_tree = create_initial_graph(LINE_DATA, initial_line_states) 
    graph_yamada = yamada.Yamada(G, n_trees=N_TREES)
    all_msts = graph_yamada.spanning_trees(initial_tree)


    lossval = net.res_line.to_numpy()
    myarray = LINE_DATA.to_numpy()
    all_msts = np.array(all_msts)
    all_open_lines, all_lines, all_loss = calculate_losses(all_msts, myarray, lossval)

    # Sort and save results
    sorted_lines = [x for _, x in sorted(zip(all_loss, all_lines))]
    sorted_open_lines = [x for _, x in sorted(zip(all_loss, all_open_lines))]
    sorted_loss = np.sort(all_loss)
    
    save_results(sorted_open_lines, sorted_lines, sorted_loss, Num_Bus)


if __name__ == "__main__":
    main()