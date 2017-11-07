import tensorflow as tf
import os
import numpy as np
import csv

class RrPeriodicalNetwork:
  ###
  # max_time: to predicting next RRI, the number of RRI provided to network
  # rnn_output_size: output size of rnn layer
  def __init__(self, max_time, rnn_output_size, isUseGPU=True, tfWriterLogPath='tf_writer', modelSavedPath='model/'):
      self.batch_size_t = tf.placeholder(tf.int32, [], name='batchSize')
      self.inputTensor = tf.placeholder(tf.float32, [None, max_time], name='inputTensor')
      self.labelTensor = tf.placeholder(tf.float32, [None], name='LabelTensor')
      self.lr = tf.placeholder(tf.float32, [], name='learningRate')

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
        
      with tf.name_scope('rnn_layer'):     
        lstmCell = tf.nn.rnn_cell.BasicLSTMCell(rnn_output_size)
        init_state = lstmCell.zero_state(self.batch_size_t, dtype=tf.float32)
        rnn_input = tf.reshape(rnn_input, [-1, max_time, 1])
        raw_output, final_state = tf.nn.dynamic_rnn(lstmCell, rnn_input, initial_state=init_state)
        self.raw_output = raw_output
        self.final_state = final_state
        outputs = tf.unstack(tf.transpose(raw_output, [1, 0, 2]))
        rnn_layer_output = outputs[-1]
        rnn_layer_output = tf.identity(rnn_layer_output, 'rnn_out')
        self.rnn_layer_output = rnn_layer_output
        #tf.summary.histogram('rnn_layer_output', rnn_layer_output)

      with tf.name_scope('expct_layer'):
        #num_hidden = 20
        #hidden= tf.layers.dense(rnn_layer_output, num_hidden, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        #tf.summary.histogram('hidden', hidden)
        expectation  = tf.layers.dense(rnn_layer_output, 1, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        expectation = (expectation + 1) * broadcast1DTensorTo2D(mean,1)
        self.expectation = tf.identity(expectation, 'E_x')
        tf.summary.histogram('predicted_e', self.expectation)

      with tf.name_scope('std_layer'):
        #num_hidden = 20
        #hidden= tf.layers.dense(rnn_layer_output, num_hidden, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        #tf.summary.histogram('hidden', hidden)
        std_from_bn_data = tf.layers.dense(rnn_layer_output, 1, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer)
        std = std_from_bn_data * std_ref
        self.std = tf.identity(std, 'STD_x')
        tf.summary.histogram('predicted_std', self.std)

      with tf.name_scope('loss'):
        # Use 1/Gauss Function as accurate part of loss function,
        # if predicted expectation == label, the loss is 0
        # if ABS(predicted expectation - label) >> std, the loss will goes infinite expotentionally
        # acc_loss_vec = 1 - tf.exp(-1*tf.pow((self.labelTensor-expectation), 2) / (2*tf.pow(std, 2) + 0.01)) 
        acc_loss_vec = tf.pow((self.labelTensor-expectation), 2) / (2*tf.pow(std, 2) + 1)
        tf.summary.histogram('acc_loss_vec', acc_loss_vec)
        loss_acc = tf.reduce_mean(acc_loss_vec)        
        tf.summary.scalar('Acc Loss', loss_acc)

        # To avoid network output std to infinite, which would cause loss_acc to 0 no matter what expectation is, we need to add penalty to std
        # the best std would be zero, the larger the worse.
        # The reference is used to normalize predicted std value      
        
        loss_std_penalty = tf.reduce_mean(tf.exp(tf.pow(std/(std_ref + 1), 2)))
        tf.summary.scalar('Std Loss', loss_std_penalty)

        self.loss = loss_acc + loss_std_penalty
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
    batch_size = 4096

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
     
      [train, log_sum, rnn_in, rnn_out, rnn_raw_out,rnn_state] = self.sess.run([self.train_op, self.tb_sum, \
                    self.rnn_input, self.rnn_layer_output, self.raw_output, self.final_state], \
                    feed_dict={self.batch_size_t: batch_size, self.inputTensor:trainInputTensor, self.labelTensor:trainLabelTensor, self.lr:learningRate})
      if np.any(np.isnan(rnn_out)):
        print "curIdx=%d"%idx
        for row in range(0, 128):
          csvRow = []
          for col in range(0,5):
            csvRow.append(str(rnn_in[row][col]))
          print ','.join(csvRow)        
        
        
      self.train_writer.add_summary(log_sum, global_step=idx)

      if int(idx/num_of_element_in_train_set) > int(prev_index / num_of_element_in_train_set):
        print "Interation count = %d"%(int(idx/num_of_element_in_train_set))
        self.train_writer.flush()
        self.validate(valData, valLabel, idx)
        self.save_model_checkpoint(int(idx/num_of_element_in_train_set))

      prev_index = idx

    return

  def inference(self, data):
    [e, std] = self.sess.run([self.expectation, self.std], feed_dict={self.inputTensor:data})
    return e,std

  def validate(self, vData, vLabel, idx):
    num_of_element = vData.shape[0]
    [e_tensor, std_tensor, loss, tb_sum] = self.sess.run([self.expectation, self.std, self.loss, self.tb_sum],\
                                                 feed_dict={self.batch_size_t: num_of_element, self.inputTensor:vData, self.labelTensor:vLabel})

    self.validation_writer.add_summary(tb_sum, global_step=idx)
    self.validation_writer.flush()
    
    vLabel = np.reshape(vLabel, (-1,1))       
    std_tensor[std_tensor < 0.1] = 0.1 # to avoid inf result, if std is 0, change it to 0.1
    relative_error_tensor = np.absolute(e_tensor - vLabel)/std_tensor
    relative_error_tensor = np.reshape(relative_error_tensor, -1)
        
    numBins = 5
    count_bin = np.zeros((numBins, 1))
    count_bin[0] = np.where(relative_error_tensor <= 0.5)[0].shape[0]
    count_bin[1] = np.where(relative_error_tensor <= 1)[0].shape[0] - count_bin[0]
    count_bin[2] = np.where(relative_error_tensor <= 2)[0].shape[0] - count_bin[0] - count_bin[1]
    count_bin[3] = np.where(relative_error_tensor <= 3)[0].shape[0] - count_bin[0] - count_bin[1] - count_bin[2]
    count_bin[4] = np.where(relative_error_tensor > 3)[0].shape[0]

    print "Total = %d. %d in [0, 0.5]std, %d in (0.5, 1.0] std, %d in (1.0, 2.0] std, %d in (2.0,3.0] std, %d in (3.0, inf] std"\
          % (num_of_element, count_bin[0], count_bin[1], count_bin[2], count_bin[3], count_bin[4])
    
    print e_tensor.shape
    print std_tensor.shape
    print vLabel.shape
    print vData.shape
    csvData = np.concatenate((e_tensor, std_tensor, vLabel, vData), axis=1)
    debugFilePathName = ('debug/validation_%d.csv'%idx)
    with open(debugFilePathName, 'wb') as oFile:
      csvWriter = csv.writer(oFile)
      csvWriter.writerow(['predicted_value', 'predicted_std', 'label', 'inference_source_1_to_N'])      
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