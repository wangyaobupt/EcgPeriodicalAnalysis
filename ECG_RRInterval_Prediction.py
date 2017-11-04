import csv
import numpy as np
from  RR_Periodical_Network import RrPeriodicalNetwork

"""
DATA Set is a list of dicts, for each dict, 3 parts are included
  data: max_time*1 float, N is number of element, max_time is the length of sequence,
  label:  float, the next element
  wid: str, a str to save waveID which is used to trace back to original waveform, this str can be empty str,
"""
def prepareDataSet(filename, max_time):
    dataSet = []
    with open(filename, 'rb') as rawFile:
        csvReader = csv.reader(rawFile)
        firstLine = True
        for row in csvReader:
            if firstLine:
                firstLine = False
                continue
            id_str = row[0]
            len = int(row[2])
            ppInterValList = row[5:]
            count = 0
            while len >  count+max_time:
                data = [float(x) for x in ppInterValList[count:count+max_time]]
                label = float(ppInterValList[count+max_time])
                dataSet.append({'data':data, 'label':label, 'wid':id_str})
                count = count + 1
    return dataSet

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
    dat_filename = './data/Stat_ECG_Peakpair_list.csv'
    max_time = 5
    val_ratio = 0.01
    learningRate =0.001
    rnn_output_size = 20
    iteration = 1000

    dataSet = prepareDataSet(dat_filename, max_time)
    train_data, train_label = convertDataSetToTensorFeed(dataSet,max_time)

    val_index_list  = randomSelectKFromN(int(val_ratio*len(dataSet)), len(dataSet))
    valSet = []
    for index in val_index_list:
      valSet.append(dataSet[index])
    val_data, val_label = convertDataSetToTensorFeed(valSet,max_time)

    network = RrPeriodicalNetwork(max_time, rnn_output_size)
    network.train(False, iteration, learningRate, train_data, train_label, val_data, val_label)





