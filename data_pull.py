# Usage
# data_pull.py -c <stock_code> -s <start_date> -e <end_date>

import tushare as ts
import os
import datetime
import sys, getopt
import pandas_datareader

DATA_DIR = "data"

class dlshare():
    def __init__(self):
        self.web = pandas_datareader.data

    def get_hist_data(self, symbol, start, end):
        if symbol.startswith("60"):
            # 上证股票
            symbol = symbol + ".ss"
        else:
            # 深圳股票
            symbol = symbol + ".sz"
        print (symbol)
        return self.web.DataReader(symbol, 'yahoo', start, end)

def _get_data(symbol='sh',start='',end=''):
    PATH = os.path.join(DATA_DIR, symbol + ".csv")
    dl = dlshare()
    if os.path.exists(PATH) and (symbol != 'ALL'):
        # the data already existed, pull the new data
        print ('The data already existed, exit.')
        return

    if start == '' or end == '':
        # pull the recent 30 days by default
        end = datetime.date.today().strftime("%Y-%m-%d")
        start = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    print("Fetch stock code:" + symbol + ", Start date:" + start + " ， End date:" + end)

    if symbol == 'ALL':
        print ("*** Will download all stock data***")
        df = ts.get_stock_basics()
        stocks = df.index.tolist()
        count = len(stocks)
        loop = 0
        for s in stocks:
            # df = dl.get_hist_data(s, start, end)
            df = ts.get_k_data(s, start, end)
            if df is None:
                print ("Warning: %s data is empty"%s)
            p = os.path.join(DATA_DIR, s + ".csv")
            loop += 1
            if loop % 10 == 0:
                print ("Download progress: %d/%d" % (loop, count))
    else:
        # df = dl.get_hist_data(symbol, start, end)
        df = ts.get_k_data(symbol, start, end)
        df.to_csv(PATH)

def main(argv):
    symbol = 'sh'
    start = ''
    end = ''
    try:
        opts, args = getopt.getopt(argv, "c:s:e", ["stock_code=", "start_date=","end_date="])
    except getopt.GetoptError:
        print ('data_pull.py -c <stock_code> -s <start_date> -e <end_date>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-c", "--stock_code"):
            symbol = arg
        elif opt in ("-s", "--start_date"):
            start = arg
        elif opt in ("-e", "--end_date"):
            end = arg

    _get_data(symbol, start, end)

if __name__ == "__main__":
    main(sys.argv[1:])