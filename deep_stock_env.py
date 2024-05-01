#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:22:13 2024

@author: pmullin
"""

import argparse
import glob, os, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class StockEnvironment:

  """
  Anything you need.  Suggestions:  __init__, train_learner, test_learner.

  I wrote train and test to just receive a learner and a dataframe as
  parameters, and train/test that learner on that data.

  You might want to print or accumulate some data as you train/test,
  so you can better understand what is happening.

  Ultimately, what you do is up to you!
  """


if __name__ == '__main__':
  # Train one Q-Learning agent for each stock in the data directory.
  # Each one will use all days in ascending order, with a train and
  # test period each day.  Each agent is NOT reset between days.
  # It is totally fine to just use one stock, and just one agent.
  # Or one agent to trade all the stocks.  Or whatever, really.

  ### Command line argument parsing.

  parser = argparse.ArgumentParser(description='Stock environment for Deep Q-Learning.')

  sim_args = parser.add_argument_group('simulation arguments')
  sim_args.add_argument('--cash', default=10000000, type=float, help='Starting cash for the agent.')
  sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--trials', default=1, type=int, help='Number of complete experimental trials.')
  sim_args.add_argument('--trips', default=1, type=int, help='Number of training trips per stock-day.')

  args = parser.parse_args()

  # Store the final in-sample and out-of-sample result of each trial.
  is_results = []
  oos_results = []

  ### HFT data file reading.

  # # Read the data only once.  It's big!
  # csv_files = glob.glob(os.path.join("..", "data", "hft_data", "*", "*_message_*.csv"))
  # date_str = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
  # stock_str = re.compile(r'([A-Z]+)_\d{4}-\d{2}-\d{2}_')

  df_list = []
  day_list = []
  sym_list = []

  # for csv_file in sorted(csv_files):
  #   date = date_str.search(csv_file)
  #   date = date.group(1)
  #   day_list.append(date)

  #   symbol = stock_str.search(csv_file)
  #   symbol = symbol.group(1)
  #   sym_list.append(symbol)

  #   # Find the order book file that matches this message file.
  #   book_file = csv_file.replace("message", "orderbook")

  #   # Read the message file and index by timestamp.
  #   df = pd.read_csv(csv_file, names=['Time','EventType','OrderID','Size','Price','Direction'])
  #   df['Time'] = pd.to_datetime(date) + pd.to_timedelta(df['Time'], unit='s')

  #   # Read the order book file and merge it with the messages.
  #   names = [f"{x}{i}" for i in range(1,11) for x in ["AP","AS","BP","BS"]]
  #   df = df.join(pd.read_csv(book_file, names=names), how='inner')
  #   df = df.set_index(['Time'])

  #   BBID_COL = df.columns.get_loc("BP1")
  #   BASK_COL = df.columns.get_loc("AP1")

  #   print (f"Read {df.shape[0]} unique order book shapshots from {csv_file}")

  #   df_list.append(df)

  # days = len(day_list)
  
  
  ## This is bas code for setting up on a much smaller data set, will
   # be incorporated into abovee code later
   
  df = pd.read_csv("./small_aapl_data.csv")
  BBID_COL = df.columns.get_loc("BP1")
  BASK_COL = df.columns.get_loc("AP1")
  
  ## Computing our state features
  df['Position'] = 0
  df['Cash'] = args.cash
  df['Stock Price'] = (df.iloc[:,BASK_COL] + df.iloc[:,BBID_COL]) / 2
  
  # Momentum-5
  df["Cumulative Returns"] = (df.loc[:, 'Stock Price'] / df.iloc[0, 3]) - 1
  df['MOM'] = df.loc[:, "Cumulative Returns"].diff(periods=5)
  df = df.drop(columns=['Stock Price'])
  df = df.fillna(0)
  
  # Imbalance
  bid_vol_cols = [x for x in range( (BBID_COL+1), (BBID_COL+40), 4 )]
  ask_vol_cols = [x for x in range( (BASK_COL+1), (BASK_COL+40), 4 )]
  df['Bid Volume'] = df.iloc[:, [bid_vol_cols]].sum(axis=1)
  df['Ask Volume'] = df.iloc[:, [ask_vol_cols]].sum(axis=1)
  df['Imbalance'] = df['Bid Volume'] - df['Ask Volume']
  df = df.drop(columns=['Bid Volume', 'Ask Volume'])
  
  # Spread
  df['Spread'] = df.iloc[:, BBID_COL] - df.iloc[:, BASK_COL]

  ## Temporary code for running on small dataset
  df_list.append(df)
  day_list.append('2024-03-08')
  sym_list.append('AAPL')
  days = 1

  

  ### Benchmark computation.

  # Compute once per day for later use.
  is_brets = []   # IS  period benchmark return
  oos_brets = []  # OOS period benchmark return

  # Prepare to receive IS and OOS cumulative returns per day,
  # potentially multiple trials.
  is_cr = [ [] for i in range(days) ]
  oos_cr = [ [] for i in range(days) ]


  ### The big learning loop.

  # Run potentially many experiments.
  for trial in range(args.trials):

    # Create an instance of the environment class.
    env = StockEnvironment( )   # TO DO: parameters.

    # TO DO: approach - train and test on a part of each day?
    for day in range(days):
      data = df_list[day]
      symbol = sym_list[day]

      # TO DO: Make a learner around here?

      if (len(is_brets) <= day):
        # Compute benchmark cumulative returns once per day only.
        
        cutoff_row = int(len(data)*0.8)

        is_start_mid = (data.iloc[0,BASK_COL] + data.iloc[0,BBID_COL]) / 2
        oos_start_mid = (data.iloc[cutoff_row,BASK_COL] + data.iloc[cutoff_row,BBID_COL]) / 2
        oos_end_mid = (data.iloc[-1,BASK_COL] + data.iloc[-1,BBID_COL]) / 2

        is_brets.append((oos_start_mid / is_start_mid) - 1.0)
        oos_brets.append((oos_end_mid / oos_start_mid) - 1.0)


      # We might do multiple trips through the data for training one stock-day.
      # Do them all at once?
      for trip in range(args.trips):

        print (f"Training {symbol}, {day_list[day]}: Trip {trip}")

        # TO DO: call  env.train_learner around here to train on one day for one trip?

        # Draw updating plot of the loss per "trip" to ensure  learner
        # was moving in the right direction.  This required exposing/returning some
        # accumulated loss values from  learner object.

        #plt.figure(1)
        #plt.clf()
        #plt.plot(losses)
        #plt.pause(0.01)  # This allows updating the plot without stopping execution.

      #plt.figure(1)
      #plt.clf()
      #plt.plot(losses)
      #plt.show()                # Stop execution to ensure not missed.
      #plt.savefig(f"some_file_name.png")    # Or this drops the final plot to some file.

      # Test the learned policy and see how it does.

      # In sample.
      print (f"In-sample {symbol}: {day_list[day]}")
      is_cr[day].append( )   # Call env.test_learner on the in-sample data.

      # Out of sample.
      print (f"Out-of-sample {symbol}: {day_list[day]}")
      oos_cr[day].append( )  # Call env.test_learner on some out-of-sample data.


  ### Print final summary stats.

  is_cr = np.array(is_cr)
  oos_cr = np.array(oos_cr)

  # Print summary results.
  print ()
  print (f"In-sample per-symbol per-day min, median, mean, max results across all {args.trials} trials")
  for day in range(days):
    print(f"IS {sym_list[day]} {day_list[day]}: {np.min(is_cr[day]):.4f}, {np.median(is_cr[day]):.4f}, \
          {np.mean(is_cr[day]):.4f}, {np.max(is_cr[day]):.4f} vs long benchmark {is_brets[day]:.4f}")

  print ()
  print (f"Out-of-sample per-symbol per-day min, median, mean, max results across all {args.trials} trials")
  for day in range(days):
    print(f"OOS {sym_list[day]} {day_list[day]}: {np.min(oos_cr[day]):.4f}, {np.median(oos_cr[day]):.4f}, \
          {np.mean(oos_cr[day]):.4f}, {np.max(oos_cr[day]):.4f} vs long benchmark {oos_brets[day]:.4f}")

