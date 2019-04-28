# Temporal Attention augmented Bilinear Networks for Financial Time-series Data Analysis

https://ieeexplore.ieee.org/document/8476227

Short description: We proposed a bilinear structure equipped with an attention mechanism that can highlight the temporal importance of each temporal event in multivariate time-series. The proposed structure is validated in the problem of predicting mid-price movements (increasing, decreasing, stationary) using Limit Order Book information. Our empirical analysis showed that the proposed structures outperformed existing models, even LSTM, CNN while being relatively fast. 

# Source Code

Code is written in python 2.7.13 with the following dependencies: 
- keras 2.1.2
- tensorflow 1.3

The naming convention follows our paper, examples how to use our code can be seen from the following files:

- example_bl.py (BL)
- example_tabl.py (TABL)

For more information, please contact thanh.tran@tuni.fi or viebboy@gmail.com

# Citation

If you use TABL in your work, please cite the following paper:

<pre>
@article{tran2018temporal,
  title={Temporal Attention-Augmented Bilinear Network for Financial Time-Series Data Analysis},
  author={Tran, Dat Thanh and Iosifidis, Alexandros and Kanniainen, Juho and Gabbouj, Moncef},
  journal={IEEE transactions on neural networks and learning systems},
  year={2018},
  publisher={IEEE}
}
</pre>
