"""
This is a data reader which reads(generate) simulation data
"""
class DataReaderSim:
  def __init__(self, size):
    self.size = size

  def prepareDataSet(self, time_steps_for_rnn):
    dataSet = []
    data = range(time_steps_for_rnn)
    label = time_steps_for_rnn
    for idx in range(self.size):
      dataSet.append({'data':data, 'label':label})
    return dataSet