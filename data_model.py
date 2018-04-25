import numpy as np
import os
import pandas as pd
import random
import time

random.seed(time.time())


class StockDataSet(object):
    def __init__(self,
                 stock_sym,
                 input_size=4,
                 num_steps=30,
                 test_ratio=0.1,
                 normalized=True,
                 close_price_only=True,
                 interval=1):
        self.stock_sym = stock_sym
        self.input_size = input_size
        self.num_steps = num_steps
        self.test_ratio = test_ratio
        self.close_price_only = close_price_only
        self.normalized = normalized

        # Read csv file
        raw_df = pd.read_csv(os.path.join("data", "%s.csv" % stock_sym))
        raw_df.columns = [x.lower() for x in raw_df.columns]
        if not 'close' in raw_df:
            raise Exception('Not valid close price in data ' + os.path.join("data", "%s.csv" % stock_sym))

        # Merge into one sequence
        if self.input_size == 4:
            # Extract features: Close price, High, Low, Volume
            raw_seq = [[raw_df['close'][i+interval], max(raw_df['high'][i:i+interval]), min(raw_df['low'][i:i + interval]), sum(raw_df['volume'][i:i + interval])] for i in range(0, len(raw_df) - interval, interval)]
            self.raw_seq = [x for y in raw_seq for x in y]
        elif self.input_size == 2:
            # Extract features: Close price, Volume
            raw_seq = [[raw_df['close'][i+interval], sum(raw_df['volume'][i:i+interval])] for i in range(0,len(raw_df)-interval,interval)]
            self.raw_seq = [x for y in raw_seq for x in y]
        elif self.input_size == 1:
            raw_df = raw_df[::interval]
            # Extract feature: Close price
            self.raw_seq = raw_df['close'].tolist()
        else:
             raise Exception('Not valid input_size:%d'%self.input_size)

        self.raw_seq = np.array(self.raw_seq)
        self.train_X, self.train_y, self.test_X, self.test_y = self._prepare_data(self.raw_seq)
        self.predict_x = self.predict_y = None

    def info(self):
        return "StockDataSet [%s] train: %d test: %d" % (
            self.stock_sym, len(self.train_X), len(self.test_y))

    def _prepare_data(self, seq):
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            if self.input_size == 1 or self.input_size == 2:
                # Close price or Close Prices, Volume
                seq = [seq[0] / seq[0] - 1.0] + [
                    curr / seq[i] - 1.0 for i, curr in enumerate(seq[1:])]
            elif self.input_size == 4:
                # Close/Close, High/Close, Low/Close, Volume/Volume
                seq = [seq[0] / seq[0] - 1.0] + \
                      [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0, curr[3] / seq[i][3] - 1.0] for i, curr in enumerate(seq[1:])]
            else:
                raise Exception('Not valid input_size:%d' % self.input_size)

        # split into groups of num_steps
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps)])
        if self.input_size > 1:
            y = np.array([np.array([seq[i + self.num_steps][0]]) for i in range(len(seq) - self.num_steps)])
        else:
            y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.test_ratio))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]
        return train_X, train_y, test_X, test_y

    def prepare_data_predict(self, seq):
        # split into items of input_size
        seq = [np.array(seq[i * self.input_size: (i + 1) * self.input_size])
               for i in range(len(seq) // self.input_size)]

        if self.normalized:
            if self.input_size == 1 or self.input_size == 2:
                # Close price or Close Prices, Volume
                seq = [seq[0] / seq[0] - 1.0] + [
                    curr / seq[i] - 1.0 for i, curr in enumerate(seq[1:])]
            elif self.input_size == 4:
                # Close/Close, High/Close, Low/Close, Volume/Volume
                seq = [seq[0] / seq[0] - 1.0] + \
                      [[curr[0] / seq[i][0] - 1.0, curr[1] / seq[i][0] - 1.0, curr[2] / seq[i][0] - 1.0, curr[3] / seq[i][3] - 1.0] for i, curr in enumerate(seq[1:])]
            else:
                raise Exception('Not valid input_size:%d' % self.input_size)

        # split into groups of num_steps, +1 to add last step
        X = np.array([seq[i: i + self.num_steps] for i in range(len(seq) - self.num_steps + 1)])
        if self.input_size > 1:
            y = np.array([np.array([seq[i + self.num_steps][0]]) for i in range(len(seq) - self.num_steps)])
        else:
            y = np.array([seq[i + self.num_steps] for i in range(len(seq) - self.num_steps)])

        # Note, the size of X is 1 bigger than Y as including last element that does not have y value
        self.predict_x = X
        self.predict_y = y
        return

    def generate_one_epoch(self, batch_size):
        num_batches = int(len(self.train_X)) // batch_size
        if batch_size * num_batches < len(self.train_X):
            num_batches += 1

        batch_indices = list(range(num_batches))
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = self.train_X[j * batch_size: (j + 1) * batch_size]
            batch_y = self.train_y[j * batch_size: (j + 1) * batch_size]
            assert set(map(len, batch_X)) == {self.num_steps}
            yield batch_X, batch_y
