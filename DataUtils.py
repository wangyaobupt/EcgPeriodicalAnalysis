import numpy as np
def splitTrainAndTestDataSet(dataSet, test_set_ratio):
  total_number = len(dataSet)
  size_of_test_set = int(test_set_ratio * total_number)
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