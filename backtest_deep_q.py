#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:02:20 2024

@author: pmullin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Run backtesting on a certain trading strategy
This is a modified version of backtest to be used with project 6
It expects a DataFrame with one column containing trade values of DIS stock
"""
def assess_strategy(trade_file = "./trades/trades.csv", starting_value = 1000000,
                    fixed_cost = 9.95, floating_cost = 0.005, trades = None):
    
    if trades is None:
        trades = pd.read_csv(trade_file)
        symbols = trades['Symbol'].unique()
        start_date = trades.iloc[0, 0]
        end_date = trades.iloc[-1, 0]
    else:
        symbols = trades.columns
        start_date = trades.index[0]
        end_date = trades.index[-1]
    stock_vals = get_data(start_date, end_date, symbols,
                          include_spy=False)
    cols = ['CASH'] + symbols.tolist()
    holdings = pd.DataFrame(np.nan, index=stock_vals.index, columns=cols)

    holdings.iloc[0, :] = 0
    holdings.loc[start_date, 'CASH'] = starting_value
    
    # trades_made = 0
    for i in range(len(holdings)):
        # trade_idx = trades_made

        # while (trade_idx < len(trades)) and (trade_idx < trades_made + 2) and (trades_made < len(trades)):
        #     if holdings.index[i] == pd.Timestamp(trades.iloc[trade_idx, 0]):
        #         holdings = execute_trade(trades.iloc[trade_idx], holdings, stock_vals,
        #                                  fixed_cost, floating_cost)
        #         trades_made += 1
        #     trade_idx += 1
            
        if trades.iloc[i, 0] != 0:
            # holdings = execute_trade(trades.iloc[i, 0], holdings, stock_vals, fixed_cost, floating_cost)
            trade = trades.iloc[i]
            cost = stock_vals.loc[trade.name, trade.index[0]] * trade.iloc[0]
            holdings.loc[trade.name, 'CASH'] -= cost
            holdings.loc[trade.name, trade.index[0]] += trade.iloc[0]
            
            
            # if trade.iloc[0] > 0:
            #     holdings.loc[trade.name, 'CASH'] -= cost
            #     holdings.loc[trade.name, trade.index[0]] += trade.iloc[0]
            # elif trade.iloc[0] < 0:
            #     holdings.loc[trade.name, 'CASH'] += cost
            #     holdings.loc[trade.name, trade.index[0]] += trade.iloc[0]
            
            holdings.loc[trade.name, 'CASH'] -= (fixed_cost + floating_cost * cost)
            
        holdings = holdings.ffill(limit=1) #We should NOT have to backfill at all
    
    daily_portfolio_values = get_data(start_date, end_date,
                                      [], include_spy=False)
    daily_portfolio_values['Daily Value'] = np.nan
    for date in holdings.index:
        daily_portfolio_values = get_daily_val(date, daily_portfolio_values,
                                               holdings, stock_vals)
    return daily_portfolio_values


"""
Get the Daily Value for a series of stock holdings
"""
def get_daily_val(date, daily_portfolio_values, holdings, stock_vals):
    daily_value = holdings.loc[date, 'CASH']
    for stock in holdings.iloc[:, 1:]:
        daily_value += (holdings.loc[date, stock] * stock_vals.loc[date, stock])
    daily_portfolio_values.loc[date] = daily_value
    return daily_portfolio_values


# """
# Update the given stock holdings
# """
# def execute_trade(trade, holdings, stock_vals, fixed_cost, floating_cost):
    
#     cost = stock_vals.loc[trade.name, trade.index[0]] * trade
#     if trade > 0:
#         holdings.loc[trade.iloc[0], 'CASH'] -= cost
#         holdings.loc[trade.iloc[0], trade.iloc[1]] += trade.iloc[3]
#     elif trade < 0:
#         holdings.loc[trade.iloc[0], 'CASH'] += cost
#         holdings.loc[trade.iloc[0], trade.iloc[1]] -= trade.iloc[3]
    
#     holdings.loc[trade.iloc[0], 'CASH'] -= (fixed_cost + floating_cost * cost)
#     return holdings


"""
Get a pandas DataFrame with only the adjusted close price of the stock we want,
and only on the dates we want and on which the market was open
"""
def get_data(start_date, end_date, symbols, column_name="Adj Close",
             include_spy=True, data_folder="./data"):
    
    date_range = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=date_range)
    filepath = data_folder + "/SPY.csv"
    new_df = pd.read_csv(filepath, index_col="Date", parse_dates=True,
                         usecols=["Date", column_name])
    df = df.join(new_df, how="inner")
    df = df.rename(columns={"Adj Close": "SPY"})
    
    
    for i in range(len(symbols)):
        stock = str(symbols[i])
        filepath = data_folder + "/" + stock + ".csv"
        new_df = pd.read_csv(filepath, index_col="Date", parse_dates=True,
                             usecols=["Date", column_name])
        df = df.join(new_df)
        df = df.rename(columns={"Adj Close": stock})
        
    if not include_spy:
        df = df.drop(columns=["SPY"])
        
    return df


"""
Test function for assess_strategy()
"""
def test_assess_strategy(trade_file="./trades/trades.csv", starting_value=1000000,
                         risk_free_rate=0.0, sample_freq=252, trades=None, baseline=None,
                         fixed_cost=9.95, floating_cost=0.005, plot=False, plot_mystrat=False,
                         print_stats=False):
    
    portfolio = assess_strategy(starting_value=starting_value, trades=trades, fixed_cost=fixed_cost,
                                floating_cost=floating_cost)
    start_date = portfolio.index[0]
    end_date = portfolio.index[-1]
    
    # if baseline is None:
    #     spx_portfolio = get_data(start_date, end_date, ['^SPX'], include_spy=False)
    # else:
    #     #Note this will not be using SPX, but the baseline specified for project 6
    #     spx_portfolio = assess_strategy(starting_value=starting_value, trades=baseline, fixed_cost=fixed_cost,
    #                                     floating_cost=floating_cost)
    
    end_value = portfolio.iloc[-1, 0]
    # spx_portfolio_end_value = spx_portfolio.iloc[-1, 0]

    portfolio["Period Returns"] = ((portfolio.iloc[:,0] / portfolio.iloc[:,0].shift()) - 1)
    # spx_portfolio["Daily Returns"] = ((spx_portfolio.iloc[:,0] / spx_portfolio.iloc[:,0].shift()) - 1)
    
    average_daily_return = portfolio.iloc[:,-1].mean()
    # spx_average_daily_return = spx_portfolio.iloc[:,-1].mean()
    
    portfolio["Portfolio Cumulative Returns"] = (portfolio.iloc[:,-2] / portfolio.iloc[0,-2]) - 1
    # spx_portfolio["Portfolio Cumulative Returns"] = (spx_portfolio.iloc[:,-2] / spx_portfolio.iloc[0,-2]) - 1

    cumulative_return = portfolio.iloc[-1,-1]
    # spx_cumulative_return = spx_portfolio.iloc[-1,-1]

    stdev_daily_return = portfolio.iloc[:,-2].std()
    # spx_stdev_daily_return = spx_portfolio.iloc[:,-2].std()

    sharpe_ratio = (((portfolio.iloc[:,-2] - risk_free_rate).mean()) / (
                    (portfolio.iloc[:,-2] - risk_free_rate).std())) * (sample_freq**0.5)
    # spx_sharpe_ratio = (((spx_portfolio.iloc[:,-2] - risk_free_rate).mean()) / (
    #                 (spx_portfolio.iloc[:,-2] - risk_free_rate).std())) * (sample_freq**0.5)
    
    # print(f'Start Date: {start_date}\nEnd Date: {end_date}\n\nPortfolio Sharpe Ratio: ' +
    #       f'{sharpe_ratio}\nEnv Sharpe Ratio: {spx_sharpe_ratio}\n\nPortfolio ADR: ' +
    #       f'{average_daily_return}\nEnv ADR: {spx_average_daily_return}\n\n' +
    #       f'Portfolio CR: {cumulative_return}\nEnv CR: {spx_cumulative_return}\n\n' +
    #       f'Portfolio SD: {stdev_daily_return}\nEnv SD: {spx_stdev_daily_return}\n\n' +
    #       f'Final Portfolio Value: {end_value}\nEnv Final Portfolio Value: {spx_portfolio_end_value}')
    
    if print_stats:
        
        print(f'Start Date: {start_date}\nEnd Date: {end_date}\n\nPortfolio Sharpe Ratio: ' +
              f'{sharpe_ratio}\nEPortfolio ADR: ' +
              f'{average_daily_return}\n' +
              f'Portfolio CR: {cumulative_return}\n' +
              f'Portfolio SD: {stdev_daily_return}\n' +
              f'Final Portfolio Value: {end_value}')
    
    if plot:
        
            
        fig, ax = plt.subplots()
        portfolio.iloc[:, 0] /= portfolio.iloc[0, 0]
        # spx_portfolio.iloc[:, 0] /= spx_portfolio.iloc[0, 0]
        # portfolio['SPX Returns'] = spx_portfolio.iloc[:, -1]
            
        portfolio[["Portfolio Cumulative Returns"]].plot(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Portfolio Returns")
        plt.legend(["Portfolio"])
        plt.show()
        
    # if plot_mystrat:
        
    #     line_idxs = []
    #     for i in range(len(trades)):
    #         if trades.iloc[i, 0] != 0:
    #             line_idxs.append(i+18289)
        
        
    #     fig, ax = plt.subplots()
    #     portfolio.iloc[:, 0] /= portfolio.iloc[0, 0]
    #     spx_portfolio.iloc[:, 0] /= spx_portfolio.iloc[0, 0]
    #     portfolio['SPX Returns'] = spx_portfolio.iloc[:, -1]
            
    #     portfolio[["Portfolio Cumulative Returns", "SPX Returns"]].plot(ax=ax)
    #     ax.set_xlabel("Date")
    #     ax.set_ylabel("Cumulative Return")
    #     ax.set_title("Portfolio vs. Baseline Cumulative Returns")
    #     ax.vlines(line_idxs, 0.3, 0.4)
    #     plt.legend(["Portfolio", "Baseline"])
    #     plt.show()
    
    
    return portfolio

