#This code is used to augment data for training of the RL agent

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name, header=0, index_col=0)

def generate_random_data(data, scale_factor=0.15):
    random_data = pd.DataFrame(np.random.normal(loc=data, scale=scale_factor * data))
    return random_data

def main():
    #Read sample 24-hour data and generate realistic data based on it
    data_folder = Path("./")
    P_DATA = load_data(data_folder / 'Data.xlsx', sheet_name='P_Data')
    Q_DATA = load_data(data_folder / 'Data.xlsx', sheet_name='Q_Data')
    
    #Read line and bus data solely to gather all data in one excel file
    LINE_DATA = load_data(data_folder / 'Data.xlsx', sheet_name='Linedata')
    BUS_DATA = load_data(data_folder / 'Data.xlsx', sheet_name='Busdata')

    #Dataframes to store generated data
    allP = pd.DataFrame()
    allQ = pd.DataFrame()
    
    #Change the number of buses and samples based on your test system
    NUM_BUSES = 33
    NUM_SAMPLES = 3001

    for i in tqdm(range(NUM_BUSES)):
        bus_i_P = pd.DataFrame()
        bus_i_Q = pd.DataFrame()
        for j in tqdm(range(NUM_SAMPLES)):
            data2P = generate_random_data(P_DATA.iloc[:, i])
            bus_i_P = pd.concat([bus_i_P, data2P], axis=0)

            data2Q = generate_random_data(Q_DATA.iloc[:, i])
            bus_i_Q = pd.concat([bus_i_Q, data2Q], axis=0)

        allP = pd.concat([allP, bus_i_P], axis=1)
        allQ = pd.concat([allQ, bus_i_Q], axis=1)

    allP.columns = P_DATA.columns.values
    allQ.columns = Q_DATA.columns.values

    writer = pd.ExcelWriter('Data_33bus.xlsx', engine='xlsxwriter')
    workbook = writer.book

    allP.to_excel(writer, sheet_name="P_Data", startrow=0, startcol=0)
    allQ.to_excel(writer, sheet_name="Q_Data", startrow=0, startcol=0)
    LINE_DATA.to_excel(writer, sheet_name="Linedata", startrow=0, startcol=0)
    BUS_DATA.to_excel(writer, sheet_name="Busdata", startrow=0, startcol=0)

    writer.close()

if __name__ == "__main__":
    main()
