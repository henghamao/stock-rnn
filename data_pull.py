# Usage
# data_pull.py -c <stock_code> -s <start_date> -e <end_date>

import tushare as ts
import os
import datetime
import sys, getopt
import pandas_datareader
import pandas as pd

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
        return self.web.DataReader(symbol, 'yahoo', start, end)

def _get_data(symbol='sh',start='',end=''):
    path = os.path.join(DATA_DIR, symbol + ".csv")
    if os.path.exists(path) and (symbol != 'ALL'):
        # the data already existed, pull the new data
        # Note: we ignored the earlier data request, in case the start date is earlier than existed data
        try:
            df_old = pd.read_csv(path)
        except:
            print ("Error to read file %s."%path)
            exit(-1)

        if not df_old.empty:
            date = df_old.tail(1)['date'].values[0]
            if date >= end:
                print ("The stock %s already existed the date %s, newer than required date %s"%(symbol, date, end))
                return
            start = (datetime.datetime.strptime(date,"%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            df = ts.get_k_data(symbol, start, end)
            df = pd.concat([df_old, df])
            df.to_csv(path,index=False)
            return

    if symbol == 'ALL':
        print ("*** Will download all stock data***")
        df = ts.get_stock_basics()
        stocks = df.index.tolist()
        count = len(stocks)
        loop = 0
        for s in stocks:
            # df = dl.get_hist_data(s, start, end)
            _get_data(s, start, end)
            loop += 1
            if loop % 10 == 0:
                print ("Download progress: %d/%d" % (loop, count))
    else:
        # df = dl.get_hist_data(symbol, start, end)
        df = ts.get_k_data(symbol, start, end)
        if df is None or df.empty:
            print("Warning: %s data is empty." % symbol)
            return
        df.to_csv(path, index=False)

def main(argv):
    symbol = 'sh'
    start = ''
    end = ''
    try:
        opts, args = getopt.getopt(argv, "c:s:e:", ["stock_code=", "start_date=","end_date="])
    except getopt.GetoptError:
        print ('Error command! data_pull.py -c <stock_code> -s <start_date> -e <end_date>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-c", "--stock_code"):
            symbol = arg
        elif opt in ("-s", "--start_date"):
            start = arg
        elif opt in ("-e", "--end_date"):
            end = arg

    if end == '':
        end = datetime.date.today().strftime("%Y-%m-%d")

    if start == '':
        # pull the recent 30 days by default
        start = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    print("Fetch stock code:" + symbol + ", Start date:" + start + " ， End date:" + end)

    _get_data(symbol, start, end)

if __name__ == "__main__":
    main(sys.argv[1:])