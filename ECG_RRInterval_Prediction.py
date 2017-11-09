import csv
import numpy as np
import os
from RR_Periodical_Network import *
from DataReaderMITBIH import DataReaderMITBIH
from DataReaderLohas import DataReaderLohas
from DataReaderSim import DataReaderSim


def splitTrainAndTestDataSet(dataSet, test_set_ratio):
    total_number = len(dataSet)
    size_of_test_set = int(test_set_ratio*total_number)
    size_of_train_set = total_number - size_of_test_set
    trainSet = dataSet[:size_of_train_set]
    testSet = dataSet[size_of_train_set:]
    return trainSet, testSet
    
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
    mit_dat_filepath = './data/20171024_ECG_MIT_peak_data/101Rpeak.txt'
    cache_data_file = './cache_data.npy'
    max_time = 50
    val_ratio = 0.01
    test_set_ratio = 0.1
    learningRate = 0.001
    rnn_output_size = 40
    iteration = 1000
    predict_range = 50
    force_read_data = True
    train_based_on_prev = False

    if (not force_read_data) and os.path.isfile(cache_data_file):
        with open(cache_data_file, "rb") as f:
            train_data, train_label, test_data, test_label, val_data, val_label = np.load(f)
    else:
        # Lohas
        #dataSet = DataReaderLohas(lohas_dat_filename).prepareDataSet(max_time)
        # MIT
        dataSet = DataReaderMITBIH(mit_dat_filepath, 360).prepareDataSet(max_time)
        # Simulation
        #dataSet = DataReaderSim(100000, 'sin_array').prepareDataSet(max_time)
        
        trainSet, testSet = splitTrainAndTestDataSet(dataSet, test_set_ratio)
        
        train_data, train_label = convertDataSetToTensorFeed(trainSet, max_time)
        test_data, test_label = convertDataSetToTensorFeed(testSet, max_time)

        val_index_list  = randomSelectKFromN(int(val_ratio*len(trainSet)), len(trainSet))
        valSet = []
        for idx in val_index_list:
            valSet.append(trainSet[idx])
        val_data, val_label = convertDataSetToTensorFeed(valSet,max_time)

        with open(cache_data_file, 'wb') as f:
            np.save(f, (train_data, train_label, test_data, test_label, val_data, val_label))

    network = RrPeriodicalNetwork(max_time, rnn_output_size, predict_range)
    network.train(train_based_on_prev, iteration, learningRate, train_data, train_label, val_data, val_label)
    #network.get_model_checkpoint()
    network.validate(test_data,test_label, 0, True, True)





