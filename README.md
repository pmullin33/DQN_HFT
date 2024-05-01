# DQN_HFT
Deep Q Learning for High Frequency Trading

Data is from lobsterdata.com - provided by Prof. Byrd

Current plans:

80/20 train/test
Use Momentum-5 as feature if possible - use midpoint as "stock price?"
Use imbalance as feature
Use spread as feature
Use BBID and BASK as feature
Need to have current position and cash as features as well
Use some form of weighted average big and weighted average ask as feature?

Basic Fully Connected FFN will pobably be sufficient to beat benchmark
7ish input - 10 - 8 - 4 - 6 - 3 outputs (buy 100, nothing, sell 100)?
