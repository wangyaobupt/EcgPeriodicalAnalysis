import csv
import numpy as np
import os
from RR_Periodical_Network import *
from DataReaderMITBIH import DataReaderMITBIH
from DataReaderLohas import DataReaderLohas
from DataReaderSim import DataReaderSim

"""
Convert dataSet into [trainData, trainLabel] for feed into tensorflow model
"""
def convertDataSetToTensorFeed(dataset,max_time):
    data = np.ndarray((len(dataset), max_time), dtype=float)
    label = np.ndarray((len(dataset)), dtype=float)

    for index in range(0, len(dataset)):
        for index_in_data in range(0,max_time):
            data[index][index_in_data] = dataset[index]['data'][index_in_data]
        label[index] = dataset[index]['label']

    return data, label

# uniform randomly select K integers from range [0,N-1]
def randomSelectKFromN(K, N):
    resultList =[]
    seqList = range(N)
    while (len(resultList) < K):
        index = (int)(np.random.rand(1)[0] * len(seqList))
        resultList.append(seqList[index])
        seqList.remove(seqList[index])
    return resultList

if __name__ == '__main__':
    lohas_dat_filename = './data/LohasData/Stat_ECG_Peakpair_list.csv'
    mit_dat_filepath = './data/20171024_ECG_MIT_peak_data/100Rpeak.txt'
    cache_data_file = './cache_data.npy'
    max_time = 50
    val_ratio = 0.01
    learningRate = 0.001
    rnn_output_size = 40
    iteration = 1000
    predict_range = 50
    force_read_data = True
    train_based_on_prev = False

    if (not force_read_data) and os.path.isfile(cache_data_file):
        with open(cache_data_file, "rb") as f:
            train_data = np.load(f)
            train_label = np.load(f)
            val_data = np.load(f)
            val_label = np.load(f)
    else:
        # Lohas
        #dataSet = DataReaderLohas(lohas_dat_filename).prepareDataSet(max_time)
        # MIT
        dataSet = DataReaderMITBIH(mit_dat_filepath, 360).prepareDataSet(max_time)
        # Simulation
        #dataSet = DataReaderSim(100000, 'sin_array').prepareDataSet(max_time)
        
        train_data, train_label = convertDataSetToTensorFeed(dataSet, max_time)

        val_index_list  = randomSelectKFromN(int(val_ratio*len(dataSet)), len(dataSet))
        valSet = []
        for index in val_index_list:
            valSet.append(dataSet[index])
        val_data, val_label = convertDataSetToTensorFeed(valSet,max_time)

        with open(cache_data_file, 'wb') as f:
            np.save(f, train_data)
            np.save(f, train_label)
            np.save(f, val_data)
            np.save(f, val_label)

    network = RrPeriodicalNetwork(max_time, rnn_output_size, predict_range)
    network.train(train_based_on_prev, iteration, learningRate, train_data, train_label, train_data, train_label)





