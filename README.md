# MCRM: Mother Compact Recurrent Memory 
## A Biologically Inspired Recurrent Neural Network Architecture

This repository contains the experiments done in the work [] by Abduallah A. Mohamed and Christian Claudel. 

MCRM is a biologically inspired RNN architecture that has a compact memory pattern. The memory pattern combines both long and short-term behaviors.

Experiments are done in PyTorch. If you find this repository helpful, please cite our work:
```
bib
```
MCRM data flow diagram:







## How to run the experiments 
There's a total of 5 experiments in this repo. 

- Adding test: python ./MCRM/adding_problem/add_test.py --model MCRM
- Copy memory test: python ./MCRM/copy_memory/copymem_test.py --model MCRM
- Sequential MNIST test: python ./MCRM/mnist_pixel/mnist_test.py --model MCRM
- Char PTB test: python ./MCRM/char_ptb/char_ptb.py --model MCRM
- Word PTB test: python ./MCRM/word_ptb/word_ptb.py --model MCRM

Our benchmark settings are drawn from [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) [repo](https://github.com/locuslab/TCN)
