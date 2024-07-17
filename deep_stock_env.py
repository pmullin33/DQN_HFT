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
import backtest_deep_q
from DeepQLearner import DeepQLearner as Learner


class StockEnvironment:

  """
  Anything you need.  Suggestions:  __init__, train_learner, test_learner.

  I wrote train and test to just receive a learner and a dataframe as
  parameters, and train/test that learner on that data.

  You might want to print or accumulate some data as you train/test,
  so you can better understand what is happening.

  Ultimately, what you do is up to you!
  """
  
  def __init__ (self, fixed = 9.95, floating = 0.005, position_limit = 1000):
      
      self.fixed_cost = fixed
      self.floating_cost = floating
      self.position_lim = position_limit
      # self.trades_made = 0
      return
      
  
  def get_state_features (self, day_data, feature_idxs, new_position):
      
      # Position MUST be first in features array
      
      state = day_data.iloc[feature_idxs]
      state = state.to_numpy()
      state[0] = new_position
      return state
  
  # def get_testing_state (self, day_idx):
  #     # ONLY FOR DEBUGGING
  #     s = np.array([day_idx + 1])
  #     return s
  
  def update_holdings (self, df, day_idx, new_position, position_idx, stock_price_idx, cash_idx, last = False, impact_costs = False):
      
      cost = (new_position - df.iloc[day_idx - 1,position_idx]) * df.iloc[day_idx,stock_price_idx]
      # if cost != 0:
      #     self.trades_made += 1
      df.iloc[day_idx, cash_idx] -= cost
      if impact_costs:
          df.iloc[day_idx, cash_idx] -= (self.fixed_cost + self.floating_cost * cost)
      df.iloc[day_idx,position_idx] = new_position
      if last:
          return
      df.iloc[day_idx + 1, cash_idx] = df.iloc[day_idx, cash_idx]
      return
      
  
  def train_learner (self, Q, df, features = ['Position', 'Best Bid/Ask Ratio']):
      
      stock_price_idx = df.columns.get_loc('Stock Price')
      position_idx = df.columns.get_loc('Position')
      cash_idx = df.columns.get_loc('Cash')
      feature_idxs = []
      for feat in features:
          feature_idxs.append(df.columns.get_loc(feat))
      
      start = self.get_state_features(df.iloc[0, :], feature_idxs, 0)
      # start = self.get_testing_state(-1)
      # print(f'Epsilon: {Q.epsilon}')
      s = start
      a = Q.test(s, allow_random=True)
      trip_loss = 0
      losses = []
      trip_rewards = []
      
      
      # Train on all the training data for this trip
      for i in range(len(df) - 1):
          
          r = df.iloc[i, stock_price_idx] * df.iloc[i, position_idx] + df.iloc[i, cash_idx]
          self.update_holdings(df, i, ((a - 1)*self.position_lim), position_idx, stock_price_idx, cash_idx)
          r = ((df.iloc[i + 1, stock_price_idx] * df.iloc[i, position_idx] + df.iloc[i, cash_idx]) / r) - 1
          s = self.get_state_features(df.iloc[i + 1,:], feature_idxs, (a - 1))
          # s = self.get_testing_state(i)

          a = Q.train(s, r)
          trip_rewards.append(r)
          loss = Q.loss
          if loss != -99999:
              losses.append(loss)
              trip_loss += loss
              Q.reset_loss()
      
      
      return trip_loss, losses, trip_rewards
  
  def test_learner (self, Q, df, features = ['Position', 'Best Bid/Ask Ratio']):
      
      self.trades_made = 0
      df_trades = pd.DataFrame(np.nan, index=df.index, columns=['Trade'])
      stock_price_idx = df.columns.get_loc('Stock Price')
      position_idx = df.columns.get_loc('Position')
      cash_idx = df.columns.get_loc('Cash')
      feature_idxs = []
      for feat in features:
          feature_idxs.append(df.columns.get_loc(feat))
      
      start = self.get_state_features(df.iloc[0, :], feature_idxs, 0)
      s = start
      a = Q.test(s)
      
      # Test the agent
      for i in range(len(df) - 1):
          
          
          prev_holding = df.iloc[i - 1, position_idx]
          self.update_holdings(df, i, ((a-1)*self.position_lim), position_idx, stock_price_idx, cash_idx)
          df_trades.iloc[i, 0] = df.iloc[i, position_idx] - prev_holding
          s = self.get_state_features(df.iloc[i+1,:], feature_idxs, (a - 1))
          a = Q.test(s)
          
      # Close any remaining position
      self.update_holdings(df, len(df) - 1, 0, position_idx, stock_price_idx, cash_idx, last=True)
      # print(f"\nTrades Made: {self.trades_made}")
      
      return df_trades



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
  sim_args.add_argument('--fixed', default=0, type=float, help='Fixed transaction cost.')
  sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
  sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
  sim_args.add_argument('--trials', default=1, type=int, help='Number of complete experimental trials.')
  sim_args.add_argument('--trips', default=10, type=int, help='Number of training trips per stock-day.')

  args = parser.parse_args()

  # Store the final in-sample and out-of-sample result of each trial.
  is_results = []
  oos_results = []

  ### HFT data file reading.

  # Read the data only once.  It's big!
  csv_files = glob.glob(os.path.join(".", "data", "hft_data", "*", "*_message_*.csv"))
  # csv_files = glob.glob(os.path.join("..", "data", "hft_data", "AAPL", "*03-04*_message_*.csv"))
  date_str = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
  stock_str = re.compile(r'([A-Z]+)_\d{4}-\d{2}-\d{2}_')

  df_list = []
  day_list = []
  sym_list = []

  for csv_file in sorted(csv_files):
    date = date_str.search(csv_file)
    date = date.group(1)
    day_list.append(date)
     
    symbol = stock_str.search(csv_file)
    symbol = symbol.group(1)
    sym_list.append(symbol)
     
    # Find the order book file that matches this message file.
    book_file = csv_file.replace("message", "orderbook")
     
    # Read the message file and index by timestamp.
    df = pd.read_csv(csv_file, names=['Time','EventType','OrderID','Size','Price','Direction'])
    df['Time'] = pd.to_datetime(date) + pd.to_timedelta(df['Time'], unit='s')
     
    # Read the order book file and merge it with the messages.
    names = [f"{x}{i}" for i in range(1,11) for x in ["AP","AS","BP","BS"]]
    df = df.join(pd.read_csv(book_file, names=names), how='inner')
    df = df.set_index(['Time'])
     
    BBID_COL = df.columns.get_loc("BP1")
    BASK_COL = df.columns.get_loc("AP1")
     
    print (f"Read {df.shape[0]} unique order book shapshots from {csv_file}")
     
    df_list.append(df) 

  days = len(day_list)
  
  
  for i in range(len(df_list)):
  
      df = df_list[i]
  ## This is bas code for setting up on a much smaller data set, will
   # be incorporated into abovee code later

  # df = pd.read_csv("./small_aapl_data.csv")
  # df = df.set_index(['Time'])
      names = [f"{x}{i}" for i in range(1,11) for x in ["AP","AS","BP","BS"]]
      df[names] /= 10000   # LOBSTER price data is multiplied by 10000, normalize it
  # BBID_COL = df.columns.get_loc("BP1")
  # BASK_COL = df.columns.get_loc("AP1")
  
  ## Computing our state features
      df['Position'] = 0
      df['Cash'] = args.cash
      df['Stock Price'] = (df.iloc[:,BASK_COL] + df.iloc[:,BBID_COL]) / 2
      
      # # Momentum-5
      # df['Price Change'] = df['Stock Price'].pct_change()
      # df['Cumulative Returns'] = (1 + df['Price Change']).rolling(window=6).apply(np.prod, raw=True) - 1
      # df['Momentum-5'] = df['Cumulative Returns'].shift(-5)
  
  # df["Cumulative Returns"] = (df.loc[:, 'Stock Price'] / df.iloc[0, df.columns.get_loc('Stock Price')]) - 1
  # df['MOM'] = df.loc[:, "Cumulative Returns"].diff(periods=5)
  # df = df.drop(columns=['Stock Price'])
      df = df.fillna(0)
  
  # Imbalance
  # bid_vol_cols = [x for x in range( (BBID_COL+1), (BBID_COL+(4*10)), 4 )]
  # ask_vol_cols = [x for x in range( (BASK_COL+1), (BASK_COL+(4*10)), 4 )]
  # df['Bid Volume'] = df.iloc[:, bid_vol_cols].sum(axis=1)
  # df['Ask Volume'] = df.iloc[:, ask_vol_cols].sum(axis=1)
  # df['Imbalance'] = df['Bid Volume'] - df['Ask Volume']
  # df = df.drop(columns=['Bid Volume', 'Ask Volume'])
  
      # Spread
      df['Spread'] = df.iloc[:, BBID_COL] - df.iloc[:, BASK_COL]
      
      # BB/BA Ratio
      df['Best Bid/Ask Ratio'] = df.iloc[:, BBID_COL] / df.iloc[:, BASK_COL]
      
      features = ['Position', 'Best Bid/Ask Ratio']
      
      columns_to_retain = features + ['BP1', 'AP1', 'Stock Price', 'Cash']
      columns_to_retain = list(set(columns_to_retain))
      df = df[columns_to_retain]
      
      # Drop consecutive duplicates based on the relevant features
      df = df.loc[(df != df.shift()).any(axis=1)]
      
      # Fill missing values (if any)
      df = df.fillna(0)
      
      df_list[i] = df

  # ## Temporary code for running on small dataset
  # df_list.append(df)
  # day_list.append('2024-03-08')
  # sym_list.append('AAPL')
  # days = 1
  # # df.to_csv('new_small_aapl_df.csv')
  

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
    env = StockEnvironment( fixed = args.fixed, floating = args.floating, position_limit = args.shares )

    # TO DO: approach - train and test on a part of each day?
    for day in range(days):
      data = df_list[day]
      symbol = sym_list[day]
      cutoff_row = int(len(data)*0.8)
      BBID_COL = data.columns.get_loc("BP1")
      BASK_COL = data.columns.get_loc("AP1")
      training_data = data.copy().iloc[:cutoff_row,:]
      is_testing_data = data.copy().iloc[:cutoff_row,:]
      oos_testing_data = data.copy().iloc[cutoff_row:,:]
      losses_over_trips = []

      # TO DO: Make a learner around here?
      learner = Learner(features=len(features), actions=3)

      if (len(is_brets) <= day):
        # Compute benchmark cumulative returns once per day only.
        
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
        
        trip_loss, losses, trip_rewards = env.train_learner(learner, training_data)
        losses_over_trips.extend(losses)
        # losses.append(trip_loss)

        # Draw updating plot of the loss per "trip" to ensure  learner
        # was moving in the right direction.  This required exposing/returning some
        # accumulated loss values from  learner object.

        plt.figure(1)
        plt.clf()
        plt.plot(losses)
        plt.pause(0.01)  # This allows updating the plot without stopping execution.

      plt.figure(1)
      plt.clf()
      plt.plot(losses_over_trips)
      plt.show()                # Stop execution to ensure not missed.
      # plt.savefig("losses.png")    # Or this drops the final plot to some file.

      # Test the learned policy and see how it does.

      # In sample.
      print (f"In-sample {symbol}: {day_list[day]}")
      trades_df = env.test_learner(learner, is_testing_data)
      is_testing_data['Portfolio Value'] = ((is_testing_data['Position'] * is_testing_data['Stock Price']) + is_testing_data['Cash'])
      # trades_eval = backtest_deep_q.test_assess_strategy(starting_value=200000, trades=trades_df, print_stats=True)
      cumulative_return = (is_testing_data.iloc[-1, -1] / is_testing_data.iloc[0, -1]) - 1
      net_return = is_testing_data.iloc[-1, -1] - is_testing_data.iloc[0, -1]
      print (f"Cumulative return: {cumulative_return}\nNet Return: {net_return}")
      is_cr[day].append(cumulative_return)   # Call env.test_learner on the in-sample data.
      is_testing_data.to_csv('test.csv')

      # Out of sample.
      trades_df = env.test_learner(learner, oos_testing_data)
      oos_testing_data['Portfolio Value'] = ((oos_testing_data['Position'] * oos_testing_data['Stock Price']) + oos_testing_data['Cash'])
      cumulative_return = (oos_testing_data.iloc[-1, -1] / oos_testing_data.iloc[0, -1]) - 1
      net_return = oos_testing_data.iloc[-1, -1] - oos_testing_data.iloc[0, -1]
      print (f"Out-of-sample {symbol}: {day_list[day]}")
      oos_cr[day].append(cumulative_return)  # Call env.test_learner on some out-of-sample data.


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

