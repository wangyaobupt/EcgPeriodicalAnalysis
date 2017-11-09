from DataReaderSim import DataReaderSim
from DataReaderMITBIH import DataReaderMITBIH
import os
from matplotlib import pyplot

mit_dat_filepath = './data/20171024_ECG_MIT_peak_data/'

if __name__ == '__main__':
  simulation_size =  1000
  train, test = DataReaderMITBIH(mit_dat_filepath,360).prepareDataSet(50, 0.2)

  print train
  print test

  #pyplot.plot(rriResult)

