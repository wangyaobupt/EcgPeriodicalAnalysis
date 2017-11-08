import tensorflow as tf
import os
import numpy as np
import csv

class RrPeriodicalNetwork:
  ###
  # max_time: to predicting next RRI, the number of RRI provided to network
  # rnn_output_size: output size of rnn layer
  def __init__(self, max_time, rnn_output_size, predict_range, isUseGPU=True, tfWriterLogPath='tf_writer', modelSavedPath='model/'):
      self.batch_size_t = tf.placeholder(tf.int32, [], name='batchSize')
      self.inputTensor = tf.placeholder(tf.float32, [None, max_time], name='inputTensor')
      self.labelTensor = tf.placeholder(tf.float32, [None], name='LabelTensor')
      self.lr = tf.placeholder(tf.float32, [], name='learningRate')
      self.predict_range = predict_range

      """
      with tf.name_scope('bn_beforeRNN'):
        mean,variance = tf.nn.moments(self.inputTensor, axes=[1])
        variance = broadcast1DTensorTo2D(variance, 1)
        std_ref = tf.sqrt(variance)
        tf.summary.histogram('std_ref_in_batch', std_ref)
        tf.summary.histogram('input_mean', mean)
        tf.summary.histogram('input_var', variance)
        broadcastMean = broadcast1DTensorTo2D(mean, max_time)
        broadcastVariance = broadcast1DTensorTo2D(variance, max_time)        
        rnn_input = tf.nn.batch_normalization(self.inputTensor, broadcastMean, broadcastVariance, \
          None, None, 1)
        self.rnn_input = rnn_input
        tf.summary.histogram('bn_output', rnn_input)
      """
        
      with tf.name_scope('rnn_layer'):     
        lstmCell = tf.nn.rnn_cell.BasicLSTMCell(rnn_output_size)
        init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
        rnn_input = tf.reshape(self.inputTensor, [-1, max_time, 1])
        raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, rnn_input, initial_state=init_state)
        self.raw_output = raw_output
        self.final_state = final_state
        outputs = tf.unstack(tf.transpose(raw_output, [1, 0, 2]))
        rnn_layer_output = outputs[-1]
        rnn_layer_output = tf.identity(rnn_layer_output, 'rnn_out')
        self.rnn_layer_output = rnn_layer_output
        tf.summary.histogram('rnn_layer_output', rnn_layer_output)

      with tf.name_scope('expct_layer'):
        #num_hidden = 20
        #hidden= tf.layers.dense(rnn_layer_output, num_hidden, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        #tf.summary.histogram('hidden', hidden)
        expectation  = tf.layers.dense(rnn_layer_output, 1, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        #expectation = (expectation + 1) * broadcast1DTensorTo2D(mean,1)
        self.expectation = tf.identity(expectation, 'E_x')
        tf.summary.histogram('predicted_e', self.expectation)

      #with tf.name_scope('std_layer'):
        #num_hidden = 20
        #hidden= tf.layers.dense(rnn_layer_output, num_hidden, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        #tf.summary.histogram('hidden', hidden)
        #std_from_bn_data = tf.layers.dense(rnn_layer_output, 1, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        #std = std_from_bn_data * std_ref
        #self.std = tf.identity(std, 'STD_x')
        #tf.summary.histogram('predicted_std', self.std)

      with tf.name_scope('loss'):
        valid_range = tf.ones([self.batch_size_t, 1])*self.predict_range
        error_tensor =  tf.abs(self.expectation - tf.reshape(self.labelTensor, [-1,1]))     
        # if error_tensor >> valid_range, classify_result  = 0
        # otherwise (error_tensor < valid_range), classify_result -> 1
        classify_result = tf.sigmoid(10*(valid_range - error_tensor)/valid_range)
        tf.summary.histogram('classify_result', classify_result)
        self.classify_result = classify_result
        
        target_tensor = tf.ones([self.batch_size_t, 1])
        self.loss = tf.losses.mean_squared_error(target_tensor, classify_result)    
        tf.summary.scalar('Loss', self.loss)

      # define train ops
      self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

      # Create session and init
      '''
      choose CPU or GPU
      '''
      if isUseGPU:
        self.sess = tf.Session()
      else:
        config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
      self.sess.run(tf.global_variables_initializer())

      # Create Tensorboard writer
      self.tb_sum = tf.summary.merge_all()
      removeFileInDir(tfWriterLogPath)
      self.train_writer = tf.summary.FileWriter(tfWriterLogPath + '/train', self.sess.graph)
      self.validation_writer = tf.summary.FileWriter(tfWriterLogPath + '/validate', self.sess.graph)

      # define saver
      all_vars = tf.global_variables()
      self.saver = tf.train.Saver(all_vars)
      self.savePath = modelSavedPath


  def setFlag(self):
    self.rnn_layer_output_isnan = True
  
  def resetFlag(self):
    self.rnn_layer_output_isnan = False 

  ###
  #   isTrainOnPreviousModel: TRUE means train model based on previous saved model files; FALSE means train from scratch
  #   iterCNT: how many iteration should be executed
  #   learningRate: currently it is designed as a fixed value
  #   trainData: training set, containing N rows, each row is an input vector which in shape (maxtime, 1)
  #   trainLabel: in shape (N,1)
  #   valData: valiadationData, in shape (M, maxtime, 1) it is recommended to be a subset of training set
  #   valLabel: in shape (M, 1)
  def train(self, isTrainOnPreviousModel, iterCNT, learningRate, trainData, trainLabel, valData, valLabel):
    num_of_element_in_train_set = trainData.shape[0]
    print "num_of_element_in_train_set=%d"%num_of_element_in_train_set
    batch_size = 256

    if isTrainOnPreviousModel:
      self.get_model_checkpoint()

    prev_index = 0
    for idx in range(0, iterCNT*num_of_element_in_train_set,batch_size):
      if idx%num_of_element_in_train_set > (idx+batch_size)%num_of_element_in_train_set:
        trainInputTensor = np.concatenate((trainData[idx%num_of_element_in_train_set:], trainData[0:(idx+batch_size)%num_of_element_in_train_set]))
        trainLabelTensor = np.concatenate((trainLabel[idx%num_of_element_in_train_set:], trainLabel[0:(idx+batch_size)%num_of_element_in_train_set]))
      else:
        trainInputTensor = trainData[idx%num_of_element_in_train_set:(idx+batch_size)%num_of_element_in_train_set]
        trainLabelTensor = trainLabel[idx%num_of_element_in_train_set:(idx+batch_size)%num_of_element_in_train_set]
     
      [train, log_sum, classify_result, loss, expectation] = self.sess.run([self.train_op, self.tb_sum, \
                    self.classify_result, self.loss, self.expectation], \
                    feed_dict={self.batch_size_t: batch_size, self.inputTensor:trainInputTensor, self.labelTensor:trainLabelTensor, self.lr:learningRate})
      """
      if np.any(np.isnan(rnn_out)):
        print "curIdx=%d"%idx
        for row in range(0, 128):
          csvRow = []
          for col in range(0,5):
            csvRow.append(str(trainInputTensor[row][col]))
          print ','.join(csvRow)
      """
      print expectation[0]
      print classify_result[0]
      print loss
        
        
      self.train_writer.add_summary(log_sum, global_step=idx)

      if int(idx/num_of_element_in_train_set) > int(prev_index / num_of_element_in_train_set):
        print "Interation count = %d"%(int(idx/num_of_element_in_train_set))
        self.train_writer.flush()
        self.validate(valData, valLabel, idx)
        self.save_model_checkpoint(int(idx/num_of_element_in_train_set))

      prev_index = idx

    return

  def inference(self, data):
    [e] = self.sess.run([self.expectation], feed_dict={self.inputTensor:data})
    return e

  def validate(self, vData, vLabel, idx):
    num_of_element = vData.shape[0]
    [e_tensor, tb_sum] = self.sess.run([self.expectation, self.tb_sum],\
                                                 feed_dict={self.batch_size_t: num_of_element, self.inputTensor:vData, self.labelTensor:vLabel})

    self.validation_writer.add_summary(tb_sum, global_step=idx)
    self.validation_writer.flush()
    
    vLabel = np.reshape(vLabel, (-1,1))       
    error_tensor = np.absolute(e_tensor - vLabel)
    error_tensor = np.reshape(error_tensor, -1)
        
    num_correct = np.where(error_tensor <= self.predict_range)[0].shape[0]
    num_wrong = num_of_element - num_correct

    print "Total = %d. %d predicted correct, acc = %f"\
          % (num_of_element, num_correct, num_correct*1.0/num_of_element)
    
    csvData = np.concatenate((e_tensor, vLabel, vData), axis=1)
    debugFilePathName = ('debug/validation_%d.csv'%idx)
    with open(debugFilePathName, 'wb') as oFile:
      csvWriter = csv.writer(oFile)
      csvWriter.writerow(['predicted_value', 'label', 'inference_source_1_to_N'])      
      csvWriter.writerows(csvData)   
    

  def save_model_checkpoint(self, i):
    self.saver.save(self.sess, self.savePath + "model.ckpt", global_step=i)
    print("Model saved as " + self.savePath + "model.ckpt")

  def get_model_checkpoint(self):
    ckpt = tf.train.get_checkpoint_state(self.savePath)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
      print("Model loaded from %s" % (ckpt.model_checkpoint_path))
      return True
    else:
      print("No checkpoint file found in " + self.my_path_to_model)
      return False

def removeFileInDir(targetDir):
  for file in os.listdir(targetDir):
    targetFile = os.path.join(targetDir, file)
    if os.path.isfile(targetFile):
      print ('Delete Old Log FIle:', targetFile)
      os.remove(targetFile)
    elif os.path.isdir(targetFile):
      print ('Delete olds in log dir: ', targetFile)
      removeFileInDir(targetFile)

def broadcast1DTensorTo2D(tensor, width_of_2D):
    hidden = tf.reshape(tensor, [-1,1])
    output = tf.tile(hidden, [1, width_of_2D])
    return output