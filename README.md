# Flexible Goal Accuracy (FGA)
This repository contains the official code for the ACL 2022 paper "Towards Fair Evaluation of Dialogue State Tracking by Flexible Incorporation of Turn-level Performances".

## Instructions to compute DST performance

Dependencies: Download MultiWOZ 2.1 dataset from https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip. Copy the data.json file to the home directory of the code repo.

1. Trade and SOM-DST: Run compute_accuracy_trade_somdst.py

2. Hi-DST: Run compute_accuracy_hi-dst

3. Trippy: Run compute_accuracy_trippy

Note: We have provided the inference result of all the four model in their respective directories. One can also generate the same by running the original codes.