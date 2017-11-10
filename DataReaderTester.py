from DataReaderSim import DataReaderSim
from DataReaderMITBIH import DataReaderMITBIH
import os
from matplotlib import pyplot

mit_dat_filepath = './data/20171024_ECG_MIT_peak_data/'

if __name__ == '__main__':
  simulation_size =  1000
  DataReaderMITBIH(mit_dat_filepath,360).dumpRRiList('./data/20171024_ECG_MIT_peak_data/rri/')

  #pyplot.plot(rriResult)

