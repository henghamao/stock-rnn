import os
import pandas as pd
import pprint
import grpc
import sys
import numpy
import threading
import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_model import StockDataSet
from model_rnn import LstmRNN
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

flags = tf.app.flags
flags.DEFINE_integer("stock_count", 100, "Stock count [100]")
flags.DEFINE_integer("input_size", 4, "Input size [2]")
flags.DEFINE_integer("num_steps", 5, "Num of steps [5]")
flags.DEFINE_integer("day_interval", 1, "Pick day interval from data[1]")
flags.DEFINE_integer("num_layers", 1, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 128, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_float("keep_prob", 0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 50, "Total training epoches. [50]")
flags.DEFINE_integer("embed_size", None, "If provided, use embedding vector of this size. [None]")
flags.DEFINE_string("stock_symbol", None, "Target stock symbol [None]")
flags.DEFINE_integer("sample_size", 4, "Number of stocks to plot during training. [4]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("train_threshold", 20, "The min sample size requirement to feed the training")
flags.DEFINE_boolean("tf_server", False, "True for using tensorflow_serving on localhost to predict")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

if not os.path.exists("logs"):
    os.mkdir("logs")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def load_stocks(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05, interval=1, train=False):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio,
                interval=interval,
                train=train)
        ]

    symbols = []
    file_black = ["_stock_list.csv","stock_list.csv", "constituents-financials.csv"]
    # Load metadata
    s = dict()
    if os.path.exists("data/stock_list.csv"):
        info = pd.read_csv("data/stock_list.csv")
        info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
        info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
        print (info['file_exists'].value_counts().to_dict())
        info = info[info['file_exists'] == True].reset_index(drop=True)
        symbols = info['symbol'].tolist()
    else:
        for root, dirs, files in os.walk("data/"):
            for file in files:
                if os.path.splitext(file)[1] == '.csv' and file not in file_black:
                    s[os.path.splitext(file)[0]] = os.path.getsize("data/"+file)
            # Order by file size (data samples)
            symbols += sorted(s.items(), key=lambda d: d[1], reverse=True)
        symbols = [symbols[i][0] for i in range(len(symbols))]

    if k is not None:
        symbols = symbols[:k]

    return [
        StockDataSet(s,
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=0.05,
                     interval=interval,
                     train=train)
        for s in symbols]

def load_sp500(input_size, num_steps, k=None, target_symbol=None, test_ratio=0.05, interval=1):
    if target_symbol is not None:
        return [
            StockDataSet(
                target_symbol,
                input_size=input_size,
                num_steps=num_steps,
                test_ratio=test_ratio,
                interval=interval)
        ]

    # Load metadata of s & p 500 stocks
    info = pd.read_csv("data/constituents-financials.csv")
    info = info.rename(columns={col: col.lower().replace(' ', '_') for col in info.columns})
    info['file_exists'] = info['symbol'].map(lambda x: os.path.exists("data/{}.csv".format(x)))
    print (info['file_exists'].value_counts().to_dict())

    info = info[info['file_exists'] == True].reset_index(drop=True)
    info = info.sort_values('market_cap', ascending=False).reset_index(drop=True)

    if k is not None:
        info = info.head(k)

    print ("Head of S&P 500 info:\n", info.head())

    # Generate embedding meta file
    info[['symbol', 'sector']].to_csv(os.path.join("logs/metadata.tsv"), sep='\t', index=False)

    return [
        StockDataSet(row['symbol'],
                     input_size=input_size,
                     num_steps=num_steps,
                     test_ratio=0.05)
        for _, row in info.iterrows()]

class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._done = 0
    self._active = 0
    self._error = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1

def _create_rpc_callback(result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(
          result_future.result().outputs['scores'].float_val)
      print("Predict result:%s"%response)
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback

def do_inference(stock_data_list, hostport='localhost:8500', concurrency=10):

  channel = grpc.insecure_channel(hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(


      channel)
  num_tests = len(stock_data_list)
  result_counter = _ResultCounter(num_tests, concurrency)
  print ("Total number of stocks to predict: %s."%num_tests)
  for label, d_ in enumerate(stock_data_list):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'stock_rnn_lstm128_step5_input4'
    request.model_spec.signature_name = 'predict_images'
    test_data = numpy.array(d_.predict_x)
    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(test_data[0], shape=[1, test_data[0].size]))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(result_counter))

  return result_counter.get_error_rate()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        rnn_model = LstmRNN(
            sess,
            FLAGS.stock_count,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            embed_size=FLAGS.embed_size,
            interval=FLAGS.day_interval
        )

        show_all_variables()

        stock_data_list = load_stocks(
            FLAGS.input_size,
            FLAGS.num_steps,
            k=FLAGS.stock_count,
            target_symbol=FLAGS.stock_symbol,
            interval=FLAGS.day_interval,
            train=FLAGS.train
        )

        if FLAGS.train:
            for s in stock_data_list[:]:
                if len(s.raw_seq) < FLAGS.train_threshold * FLAGS.num_steps:
                    stock_data_list.remove(s)
                    print ("Info: %s sample is too small, remove from training"%s.stock_sym)
            if stock_data_list.__len__() == 0:
                print ("No data to train.")
                exit(-1)
            rnn_model.train(stock_data_list, FLAGS)
        else:
            for s in stock_data_list[:]:
                if len(s.raw_seq) < FLAGS.num_steps:
                    stock_data_list.remove(s)
                    print ("Info: %s sample is too small, less than num_step"%s.stock_sym)
            if stock_data_list.__len__() == 0:
                print ("No data to predict.")
                exit(-1)
            if FLAGS.tf_server:
                print("Submit requests to local tensorflow_serving.")
                do_inference(stock_data_list)
            else:
                rnn_model.predict(stock_data_list, FLAGS)


if __name__ == '__main__':
    tf.app.run()
