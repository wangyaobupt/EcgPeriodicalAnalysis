import csv

class DataReaderLohas:
  def __init__(self, filename):
    self.filename = filename

  """
  DATA Set is a list of dicts, for each dict, 3 parts are included
    data: time_steps_for_rnn*1 float, N is number of element, max_time is the length of sequence,
    label:  float, the next element
    wid: str, a str to save waveID which is used to trace back to original waveform, this str can be empty str, ONLY for debug purpose
  """
  def prepareDataSet(self, time_steps_for_rnn):
      dataSet = []
      with open(self.filename, 'rb') as rawFile:
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
              while len >  count+time_steps_for_rnn:
                  data = [float(x) for x in ppInterValList[count:count+time_steps_for_rnn]]
                  label = float(ppInterValList[count+time_steps_for_rnn])
                  dataSet.append({'data':data, 'label':label, 'wid':id_str})
                  count = count + 1
      return dataSet