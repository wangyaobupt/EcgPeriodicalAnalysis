import math

"""
This is a data reader which reads(generate) simulation data
"""
class DataReaderSim:
  def __init__(self, size, t):
    self.size = size
    self.t = t

  def prepareDataSet(self, time_steps_for_rnn):
    dataSet = []
    if self.t == 'simple_same':
      data = range(time_steps_for_rnn)
      label = time_steps_for_rnn
      for idx in range(self.size):
        dataSet.append({'data':data, 'label':label})
    elif self.t == 'simple_one_array':
      for idx in range(self.size):
        data = range(idx, idx+time_steps_for_rnn)
        label = idx+time_steps_for_rnn
        dataSet.append({'data':data, 'label':label})
    elif self.t == 'sin_array':
      x = [math.sin(2*math.pi*idx*0.1) for idx in range(self.size + time_steps_for_rnn)]
      for idx in range(self.size):
        data = x[idx:idx+time_steps_for_rnn]
        label = x[idx+time_steps_for_rnn]
        dataSet.append({'data':data, 'label':label})
      
    return dataSet