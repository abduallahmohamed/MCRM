# MCRM: Mother Compact Recurrent Memory 
## A Biologically Inspired Recurrent Neural Network Architecture

This repository contains the experiments in [MCRM: Mother Compact Recurrent Memory
A Biologically Inspired Recurrent Neural Network Architecture](https://arxiv.org/pdf/1808.02016.pdf) work by Abduallah Mohamed and Christian Claudel. 

MCRM is a biologically inspired RNN architecture that has a compact memory pattern. The memory pattern combines both long and short-term behaviors.

```

@ARTICLE{2018arXiv180802016M,
   author = {{Mohamed}, A. and {Claudel}, C.},
    title = "{MCRM: Mother Compact Recurrent Memory A Biologically Inspired Recurrent Neural Network Architecture}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1808.02016},
 keywords = {Computer Science - Neural and Evolutionary Computing, Computer Science - Machine Learning, Statistics - Machine Learning},
     year = 2018,
    month = aug,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180802016M},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

```
## MCRM data flow diagram:
![MCRM Data flow](MCRM.bmp?raw=true "Title")

## Supplied implementation 
- `class MCRMCell` which is MCRM cell . 
- `class MCRM` which is a generic class to utilizie multiple MCRM cells. It has the same Pytorch RNN modules signatures.
both classes are available [here](/MCRM/mcrm.py)
## How to run the experiments 
There's a total of 5 experiments in this repo. 

- **Adding test**: `python ./MCRM/adding_problem/add_test.py --model MCRM`
- **Copy memory test**: `python ./MCRM/copy_memory/copymem_test.py --model MCRM`
- **Sequential MNIST test**: `python ./MCRM/mnist_pixel/mnist_test.py --model MCRM`
- **Char PTB test**: `python ./MCRM/char_ptb/char_ptb.py --model MCRM`
- **Word PTB test**: `python ./MCRM/word_ptb/word_ptb.py --model MCRM`

Experiments are done in PyTorch (0.4.1) using Python 3.6. 

Our benchmark settings are drawn from [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) [repo](https://github.com/locuslab/TCN)
