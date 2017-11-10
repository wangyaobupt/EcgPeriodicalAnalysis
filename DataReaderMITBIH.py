import os
from DataUtils import *

class DataReaderMITBIH:
  def __init__(self, filename, sampleRate):
    self.filename = filename
    self.sampleRate = sampleRate

  def readRriListFromFile(self, filename):
    with open(filename, 'r') as f:
      r_peak_pos_list = f.read().split(",")
    r_peak_pos_list = [x for x in r_peak_pos_list if len(x.strip()) > 0]
    r_peak_pos_list = map(int, r_peak_pos_list)

    r_peak_pos_in_ms_list = []
    for pos in r_peak_pos_list:
      r_peak_pos_in_ms_list.append(pos*1000.0/self.sampleRate)

    rri_list = []
    for idx in range(0, len(r_peak_pos_in_ms_list)-1):
      rri_list.append(r_peak_pos_in_ms_list[idx+1] - r_peak_pos_in_ms_list[idx])

    return rri_list

  def prepareDataSet(self, time_steps_for_rnn, testSetRatio=0.1):
    if os.path.isfile(self.filename):
      result = self.readDataSetFromOneFile(self.filename, time_steps_for_rnn)
      trainSet, testSet = splitTrainAndTestDataSet(result, testSetRatio)
    elif os.path.isdir(self.filename):
      trainSet = []
      testSet = []
      for root, dirs, filenames in os.walk(self.filename):
        for file in filenames:
          if 'Rpeak.txt' in file:
            trainIter, testIter= splitTrainAndTestDataSet(self.readDataSetFromOneFile(os.path.join(root, file), time_steps_for_rnn), testSetRatio)
            trainSet += trainIter
            testSet += testIter

    return trainSet, testSet

  def readDataSetFromOneFile(self, filename, time_steps_for_rnn):
    dataSet = []
    rriList = self.readRriListFromFile(filename)
    idx = 0
    while (idx + time_steps_for_rnn) < len(rriList):
      dataList = [x for x in rriList[idx:idx + time_steps_for_rnn]]
      label = rriList[idx + time_steps_for_rnn]
      dataSet.append({'data': dataList, 'label': label})
      idx = idx + 1

    return dataSet

  def dumpRRiList(self, folderPath):
    if os.path.isfile(self.filename):
      rri = self.readRriListFromFile(self.filename)
      with open(os.path.join(folderPath, self.filename+'_rri.txt'), 'w') as f:
        f.write(rri)
    elif os.path.isdir(self.filename):
      for root, dirs, filenames in os.walk(self.filename):
        for file in filenames:
          if 'Rpeak.txt' in file:
            rri = self.readRriListFromFile(os.path.join(root,file))
            with open(os.path.join(folderPath, file + '_rri.txt'), 'w') as f:
              f.write(",".join(map(str, rri)))

