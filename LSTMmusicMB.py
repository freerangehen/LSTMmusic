############## music generation with 3 layer LSTM ######################
#
#           Distributed under the MIT license by Henry Ip
########################################################################


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math 
import copy as cp
from copy import deepcopy
import pickle as pkl
import time
import os
import theano #typical 2x speed up for small network, 400x600x600 net ~6x speed up
from theano import tensor as T, function, printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sandbox.cuda as cuda
from random import shuffle
#from theano.compile.nanguardmode import NanGuardMode


# midi utility scripts should be installed in ./midi/utils.py
import midi.MidiOutStream 
import midi.MidiInStream
import midi.MidiInFile
import midi.MidiToText
from midi.utils import midiread 
from midi.utils import midiwrite

# set plot sizes if this code is copied to ipython/jupyter notebook
import pylab
pylab.rcParams['figure.figsize'] = (20, 5)
np.set_printoptions(threshold='nan')


class RNN4Music:
    def __init__(self,h1_length = 200, h2_length = 200, h3_length = 200, io_length=88, 
                 R1=0.0001, R2=0.0001, R3=0.0001, Rout=0.0001):
        '''
        Constructor of RNN4Music objects. Implements a fixed 3 layer LSTM. For architecture, see Alex Graves' paper: 
        [generated sequences with recurrent neural networks](http://arxiv.org/pdf/1308.0850v5.pdf)
    

        arguments:

        h1_length (int): no. of LSTM units in first layer 

        h2_length (int): no. of LSTM units in second layer 

        h3_length (int): third layer size (output contributes to overall output only) 	

        io_length (int): no. of inputs. 

        R1 (int): training rate of weights and biases in layer 1

        R2 (int): training rate of weights and biases in layer 2

        R3 (int): training rate of weights and biases in layer 3

        Rout (int): training rate of weights and biases in output layer
        '''

        # computational parameters for safe-guarding:
        self.epsilon = np.float32(0.00001) #small number to guard against dividing by zero
        self.gradClip = np.float32(30.0) # gradient cap parameter, cap too much it oscillates, cannot exceed the randomness of the weights... 

        # parameters for randomly initialising weights/biases
        self.muRand = np.float32(0.01) # for weights/biases that are initialisaed to ~uniform(0, muRand) 
        self.sigRand = np.float32(0.1) #S.D. of normally distributed initialisation of weights/biases

        # house keeping variables        
        self.parameterSaved = False
        self.parameterLoaded = False
        self.instantaneousCost = []        
        
        #push arguments as instance variables for network parameters
        self.h1_length = h1_length
        self.h2_length = h2_length
        self.h3_length = h3_length
        self.io_length = io_length
        self.h1io_l = h1_length*io_length
        self.h2io_l = (io_length+h1_length)*h2_length
        self.h3io_l = (io_length + h1_length + h2_length)*h3_length
        self.out_l = (h2_length+h1_length+h3_length)*io_length
        self.R1 = R1; self.R2 = R2; self.R3 = R3; self.Rout = Rout

        self.T_rng = RandomStreams()
        
        self.loss = theano.shared(value=np.float32(0.0), name='loss', borrow=True, allow_downcast=True)



        ###### network input variable #####

        self.xt = theano.shared(value=np.float32(np.random.uniform(np.float(0.0), np.float32(1.0), io_length)), name='xt', borrow=True, allow_downcast=True)
        
        
                
        ###### weights/biases and variables for first layer ###### 
               
        #input gate:
        self.It_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='It_1', borrow=True, allow_downcast=True)
        self.Itd1_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='Itd1_1', borrow=True, allow_downcast=True)
        self.it_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='it_1', borrow=True, allow_downcast=True)
        self.Wxi_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h1_length).reshape((h1_length,io_length))), name='Wxi_1', borrow=True, allow_downcast=True)
        self.Whi_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h1_length).reshape((h1_length,h1_length))), name='Whi_1', borrow=True, allow_downcast=True)
        self.Wci_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length)), name='Wci_1', borrow=True, allow_downcast=True)
        self.bi_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h1_length,self.sigRand,h1_length)), name='bi_1', borrow=True, allow_downcast=True)
        #forget gate:
        self.Ft_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='Ft_1', borrow=True, allow_downcast=True)
        self.Ftd1_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='Ftd1_1', borrow=True, allow_downcast=True)
        self.ft_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='ft_1', borrow=True, allow_downcast=True)
        self.Wxf_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h1_length).reshape((h1_length,io_length))), name='Wxf_1', borrow=True, allow_downcast=True)
        self.Whf_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h1_length).reshape((h1_length,h1_length))), name='Whf_1', borrow=True, allow_downcast=True)
        self.Wcf_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length)), name='Wcf_1', borrow=True, allow_downcast=True)
        self.bf_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h1_length,self.sigRand,h1_length)), name='bf_1', borrow=True, allow_downcast=True)
        #output gate:
        self.Ot_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='Ot_1', borrow=True, allow_downcast=True)
        self.Otd1_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='Otd1_1', borrow=True, allow_downcast=True)
        self.ot_1 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h1_length)), name='ot_1', borrow=True, allow_downcast=True)
        self.Wxo_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h1_length).reshape((h1_length,io_length))), name='Wxo_1', borrow=True, allow_downcast=True)
        self.Who_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h1_length).reshape((h1_length,h1_length))), name='Who_1', borrow=True, allow_downcast=True)
        self.Wco_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length)), name='Wco_1', borrow=True, allow_downcast=True)
        self.bo_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h1_length,self.sigRand,h1_length)), name='bo_1', borrow=True, allow_downcast=True)
        #state:
        self.Ct_1 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0), self.muRand, h1_length)), name='Ct_1', borrow=True, allow_downcast=True)
        self.ctd1_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length)), name='ctd1_1', borrow=True, allow_downcast=True)
        self.ct_1 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0), self.muRand, h1_length)), name='ct_1', borrow=True, allow_downcast=True)
        self.Wxc_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h1_length).reshape((h1_length,io_length))), name='Wxc_1', borrow=True, allow_downcast=True)
        self.Whc_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h1_length).reshape((h1_length,h1_length))), name='Whc_1', borrow=True, allow_downcast=True)
        self.bc_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length)), name='bc_1', borrow=True, allow_downcast=True)
        #output:
        self.ht_1 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, h1_length)), name='ht_1', borrow=True, allow_downcast=True)
        self.htd1_1 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length)), name='htd1_1', borrow=True, allow_downcast=True)
        
       
        ###### weights/biases and variables for second layer ###### 
               
        #input gate:
        self.It_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='It_2', borrow=True, allow_downcast=True)
        self.Itd1_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='Itd1_2', borrow=True, allow_downcast=True)
        self.it_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='it_2', borrow=True, allow_downcast=True)
        self.Wxi_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h2_length).reshape((h2_length,io_length))), name='Wxi_2', borrow=True, allow_downcast=True)
        self.Wxj_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h2_length).reshape((h2_length,h1_length))), name='Wxj_2', borrow=True, allow_downcast=True)
        self.Whi_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h2_length).reshape((h2_length,h2_length))), name='Whi_2', borrow=True, allow_downcast=True)
        self.Wci_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length)), name='Wci_2', borrow=True, allow_downcast=True)
        self.bi_2= theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h2_length,self.sigRand,h2_length)), name='bi_2', borrow=True, allow_downcast=True)
        #forget gate:
        self.Ft_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='Ft_2', borrow=True, allow_downcast=True)
        self.Ftd1_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='Ftd1_2', borrow=True, allow_downcast=True)
        self.ft_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='ft_2', borrow=True, allow_downcast=True)
        self.Wxf_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h2_length).reshape((h2_length,io_length))), name='Wxf_2', borrow=True, allow_downcast=True)
        self.Wfj_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h2_length).reshape((h2_length,h1_length))), name='Wfj_2', borrow=True, allow_downcast=True)
        self.Whf_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h2_length).reshape((h2_length,h2_length))), name='Whf_2', borrow=True, allow_downcast=True)
        self.Wcf_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length)), name='Wcf_2', borrow=True, allow_downcast=True)
        self.bf_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h2_length,self.sigRand,h2_length)), name='bf_2', borrow=True, allow_downcast=True)
        #output gate:
        self.Ot_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='Ot_2', borrow=True, allow_downcast=True)
        self.Otd1_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='Otd1_2', borrow=True, allow_downcast=True)
        self.ot_2 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h2_length)), name='ot_2', borrow=True, allow_downcast=True)
        self.Wxo_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h2_length).reshape((h2_length,io_length))), name='Wxo_2', borrow=True, allow_downcast=True)
        self.Woj_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h2_length).reshape((h2_length,h1_length))), name='Woj_2', borrow=True, allow_downcast=True)
        self.Who_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h2_length).reshape((h2_length,h2_length))), name='Who_2', borrow=True, allow_downcast=True)
        self.Wco_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length)), name='Wco_2', borrow=True, allow_downcast=True)
        self.bo_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h2_length,self.sigRand,h2_length)), name='bo_2', borrow=True, allow_downcast=True)
        #state:
        self.Ct_2 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, h2_length)), name='Ct_2', borrow=True, allow_downcast=True)
        self.ctd1_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length)), name='ctd1_2', borrow=True, allow_downcast=True)
        self.ct_2 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, h2_length)), name='ct_2', borrow=True, allow_downcast=True)
        self.Wxc_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h2_length).reshape((h2_length,io_length))), name='Wxc_2', borrow=True, allow_downcast=True)
        self.Wcj_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h1_length*h2_length).reshape((h2_length,h1_length))), name='Wcj_2', borrow=True, allow_downcast=True)
        self.Whc_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h2_length).reshape((h2_length,h2_length))), name='Whc_2', borrow=True, allow_downcast=True)
        self.bc_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length)), name='bc_2', borrow=True, allow_downcast=True)
        #output:
        self.ht_2 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, h2_length)), name='ht_2', borrow=True, allow_downcast=True)
        self.htd1_2 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length)), name='htd1_2', borrow=True, allow_downcast=True)
        

        ###### weights/biases and variables for third layer ######
               
        #input gate:
        self.It_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='It_3', borrow=True, allow_downcast=True)
        self.Itd1_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='Itd1_3', borrow=True, allow_downcast=True)
        self.it_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='it_3', borrow=True, allow_downcast=True)
        self.Wxi_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h3_length).reshape((h3_length,io_length))), name='Wxi_3', borrow=True, allow_downcast=True)
        self.Wxj_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h3_length).reshape((h3_length,h2_length))), name='Wxj_3', borrow=True, allow_downcast=True)
        self.Whi_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length*h3_length).reshape((h3_length,h3_length))), name='Whi_3', borrow=True, allow_downcast=True)
        self.Wci_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length)), name='Wci_3', borrow=True, allow_downcast=True)
        self.bi_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h3_length,self.sigRand,h3_length)), name='bi_3', borrow=True, allow_downcast=True)
        #forget gate:
        self.Ft_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='Ft_3', borrow=True, allow_downcast=True)
        self.Ftd1_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='Ftd1_3', borrow=True, allow_downcast=True)
        self.ft_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='ft_3', borrow=True, allow_downcast=True)
        self.Wxf_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h3_length).reshape((h3_length,io_length))), name='Wxf_3', borrow=True, allow_downcast=True)
        self.Wfj_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h3_length).reshape((h3_length,h2_length))), name='Wfj_3', borrow=True, allow_downcast=True)
        self.Whf_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length*h3_length).reshape((h3_length,h3_length))), name='Whf_3', borrow=True, allow_downcast=True)
        self.Wcf_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length)), name='Wcf_3', borrow=True, allow_downcast=True)
        self.bf_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h3_length,self.sigRand,h3_length)), name='bf_3', borrow=True, allow_downcast=True)
        #output gate:
        self.Ot_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='Ot_3', borrow=True, allow_downcast=True)
        self.Otd1_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='Otd1_3', borrow=True, allow_downcast=True)
        self.ot_3 = theano.shared(value=np.float32(np.random.uniform(np.float32(0.0), self.muRand, h3_length)), name='ot_3', borrow=True, allow_downcast=True)
        self.Wxo_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h3_length).reshape((h3_length,io_length))), name='Wxo_3', borrow=True, allow_downcast=True)
        self.Woj_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h3_length).reshape((h3_length,h2_length))), name='Woj_3', borrow=True, allow_downcast=True)
        self.Who_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length*h3_length).reshape((h3_length,h3_length))), name='Who_3', borrow=True, allow_downcast=True)
        self.Wco_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length)), name='Wco_3', borrow=True, allow_downcast=True)
        self.bo_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/h3_length,self.sigRand,h3_length)), name='bo_3', borrow=True, allow_downcast=True)
        #state:
        self.Ct_3 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, h3_length)), name='Ct_3', borrow=True, allow_downcast=True)
        self.ctd1_3 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length)), name='ctd1_3', borrow=True, allow_downcast=True)
        self.ct_3 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, h3_length)), name='ct_3', borrow=True, allow_downcast=True)
        self.Wxc_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h3_length).reshape((h3_length,io_length))), name='Wxc_3', borrow=True, allow_downcast=True)
        self.Wcj_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h2_length*h3_length).reshape((h3_length,h2_length))), name='Wcj_3', borrow=True, allow_downcast=True)
        self.Whc_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length*h3_length).reshape((h3_length,h3_length))), name='Whc_3', borrow=True, allow_downcast=True)
        self.bc_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length)), name='bc_3', borrow=True, allow_downcast=True)
        #output:
        self.ht_3 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, h3_length)), name='ht_3', borrow=True, allow_downcast=True)
        self.htd1_3 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand,h3_length)), name='htd1_3', borrow=True, allow_downcast=True)
        
        
        ###### weights/biases and variables for network output layer ######
        self.Yt = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, io_length)), name='Yt', borrow=True, allow_downcast=True)
        self.Ytd1 = theano.shared(value=np.float32(np.random.normal(np.float32(0.0),self.sigRand, io_length)), name='Ytd1', borrow=True, allow_downcast=True)
        self.yt = theano.shared(value=np.float32(np.random.uniform(-self.muRand, self.muRand, io_length)), name='yt', borrow=True, allow_downcast=True)
        self.Why_1 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h1_length).reshape((io_length,h1_length))), name='Why_1', borrow=True, allow_downcast=True)
        self.Why_2 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h2_length).reshape((io_length,h2_length))), name='Why_2', borrow=True, allow_downcast=True)
        self.Why_3 = theano.shared(value = np.float32(np.random.normal(np.float32(0.0),self.sigRand,io_length*h3_length).reshape((io_length,h3_length))), name='Why_3', borrow=True, allow_downcast=True)
        self.by = theano.shared(value = np.float32(np.random.normal(np.float32(0.5)/io_length,self.sigRand,io_length)), name='by', borrow=True, allow_downcast=True)
        
        
       
        
        ###### first layer RMSprop previous gradients ###### 
        #input gate:
        self.DWxi1p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h1_length)]), name='DWxi1p', borrow=True, allow_downcast=True)
        self.DWhi1p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]), name='DWhi1p', borrow=True, allow_downcast=True)
        self.DWci1p = theano.shared(value = np.float32([0.0]*h1_length), name='DWci1p', borrow=True, allow_downcast=True)
        self.Dbi1p = theano.shared(value = np.float32([0.0]*h1_length), name='Dbi1p', borrow=True, allow_downcast=True)
        #forget gate:
        self.DWxf1p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h1_length)]), name='DWxf1p', borrow=True, allow_downcast=True)
        self.DWhf1p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]), name='DWhf1p', borrow=True, allow_downcast=True)
        self.DWcf1p = theano.shared(value = np.float32([0.0]*h1_length), name='DWcf1p', borrow=True, allow_downcast=True)
        self.Dbf1p = theano.shared(value = np.float32([0.0]*h1_length), name='Dbf1p', borrow=True, allow_downcast=True)
        #output gate:
        self.DWxo1p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h1_length)]), name='DWxo1p', borrow=True, allow_downcast=True)
        self.DWho1p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]), name='DWho1p', borrow=True, allow_downcast=True)
        self.DWco1p = theano.shared(value = np.float32([0.0]*h1_length), name='DWco1p', borrow=True, allow_downcast=True)
        self.Dbo1p = theano.shared(value = np.float32([0.0]*h1_length), name='Dbo1p', borrow=True, allow_downcast=True)
        #state:
        self.DWxc1p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h1_length)]), name='DWxc1p', borrow=True, allow_downcast=True)
        self.DWhc1p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]), name='DWhc1p', borrow=True, allow_downcast=True)
        self.Dbc1p = theano.shared(value = np.float32([0.0]*h1_length), name='Dbc1p', borrow=True, allow_downcast=True)
        ###### second layer RMSprop previous gradients ###### 
        #input gate:
        self.DWxi2p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h2_length)]), name='DWxi2p', borrow=True, allow_downcast=True)
        self.DWxj2p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]), name='DWxj2p', borrow=True, allow_downcast=True)
        self.DWhi2p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]), name='DWhi2p', borrow=True, allow_downcast=True)
        self.DWci2p = theano.shared(value = np.float32([0.0]*h2_length), name='DWci2p', borrow=True, allow_downcast=True)
        self.Dbi2p = theano.shared(value = np.float32([0.0]*h2_length), name='Dbi2p', borrow=True, allow_downcast=True)
        #forget gate:
        self.DWxf2p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h2_length)]), name='DWxf2p', borrow=True, allow_downcast=True)
        self.DWfj2p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]), name='DWfj2p', borrow=True, allow_downcast=True)
        self.DWhf2p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]), name='DWhf2p', borrow=True, allow_downcast=True)
        self.DWcf2p = theano.shared(value = np.float32([0.0]*h2_length), name='DWcf2p', borrow=True, allow_downcast=True)
        self.Dbf2p = theano.shared(value = np.float32([0.0]*h2_length), name='Dbf2p', borrow=True, allow_downcast=True)
        #output gate:
        self.DWxo2p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h2_length)]), name='DWxo2p', borrow=True, allow_downcast=True)
        self.DWoj2p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]), name='DWoj2p', borrow=True, allow_downcast=True)
        self.DWho2p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]), name='DWho2p', borrow=True, allow_downcast=True)
        self.DWco2p = theano.shared(value = np.float32([0.0]*h2_length), name='DWco2p', borrow=True, allow_downcast=True)
        self.Dbo2p = theano.shared(value = np.float32([0.0]*h2_length), name='Dbo2p', borrow=True, allow_downcast=True)
        #state:
        self.DWxc2p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h2_length)]), name='DWxc2p', borrow=True, allow_downcast=True)
        self.DWcj2p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]), name='DWcj2p', borrow=True, allow_downcast=True)
        self.DWhc2p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]), name='DWhc2p', borrow=True, allow_downcast=True)
        self.Dbc2p = theano.shared(value = np.float32([0.0]*h2_length), name='Dbc2p', borrow=True, allow_downcast=True)
        ###### third layer RMSprop previous gradients ###### 
        #input gate:
        self.DWxi3p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h3_length)]), name='DWxi3p', borrow=True, allow_downcast=True)
        self.DWxj3p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]), name='DWxj3p', borrow=True, allow_downcast=True)
        self.DWhi3p = theano.shared(value = np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]), name='DWhi3p', borrow=True, allow_downcast=True)
        self.DWci3p = theano.shared(value = np.float32([0.0]*h3_length), name='DWci3p', borrow=True, allow_downcast=True)
        self.Dbi3p = theano.shared(value = np.float32([0.0]*h3_length), name='Dbi3p', borrow=True, allow_downcast=True)
        #forget gate:
        self.DWxf3p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h3_length)]), name='DWxf3p', borrow=True, allow_downcast=True)
        self.DWfj3p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]), name='DWfj3p', borrow=True, allow_downcast=True)
        self.DWhf3p = theano.shared(value = np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]), name='DWhf3p', borrow=True, allow_downcast=True)
        self.DWcf3p = theano.shared(value = np.float32([0.0]*h3_length), name='DWcf3p', borrow=True, allow_downcast=True)
        self.Dbf3p = theano.shared(value = np.float32([0.0]*h3_length), name='Dbf3p', borrow=True, allow_downcast=True)
        #output gate:
        self.DWxo3p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h3_length)]), name='DWxo3p', borrow=True, allow_downcast=True)
        self.DWoj3p = theano.shared(value = np.float32([[0.0 for m in range(self.h2_length)] for n in range(self.h3_length)]), name='DWoj3p', borrow=True, allow_downcast=True)
        self.DWho3p = theano.shared(value = np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]), name='DWho3p', borrow=True, allow_downcast=True)
        self.DWco3p = theano.shared(value = np.float32([0.0]*h3_length), name='DWco3p', borrow=True, allow_downcast=True)
        self.Dbo3p = theano.shared(value = np.float32([0.0]*h3_length), name='Dbo3p', borrow=True, allow_downcast=True)
        #state:
        self.DWxc3p = theano.shared(value = np.float32([[0.0]*self.io_length for n in range(self.h3_length)]), name='DWxc3p', borrow=True, allow_downcast=True)
        self.DWcj3p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]), name='DWcj3p', borrow=True, allow_downcast=True)
        self.DWhc3p = theano.shared(value = np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]), name='DWhc3p', borrow=True, allow_downcast=True)
        self.Dbc3p = theano.shared(value = np.float32([0.0]*h3_length), name='Dbc3p', borrow=True, allow_downcast=True)
        ###### output layer RMSprop previous gradients ###### 
        self.DWhy1p = theano.shared(value = np.float32([[0.0]*self.h1_length for n in range(self.io_length)]), name='DWhy1p', borrow=True, allow_downcast=True)
        self.DWhy2p = theano.shared(value = np.float32([[0.0]*self.h2_length for n in range(self.io_length)]), name='DWhy2p', borrow=True, allow_downcast=True)
        self.DWhy3p = theano.shared(value = np.float32([[0.0]*self.h3_length for n in range(self.io_length)]), name='DWhy3p', borrow=True, allow_downcast=True)
        self.Dbyp = theano.shared(value = np.float32([0.0]*io_length), name='Dbyp', borrow=True, allow_downcast=True)
     
    
    def resetRMSgrads(self):
     
        ###### first layer RMSprop previous gradients ###### 
        #input gate:
        self.DWxi1p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h1_length)]))
        self.DWhi1p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
        self.DWci1p.set_value(np.float32([0.0]*self.h1_length))
        self.Dbi1p.set_value(np.float32([0.0]*self.h1_length))
        #forget gate:
        self.DWxf1p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h1_length)]))
        self.DWhf1p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
        self.DWcf1p.set_value(np.float32([0.0]*self.h1_length))
        self.Dbf1p.set_value(np.float32([0.0]*self.h1_length))
        #output gate:
        self.DWxo1p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h1_length)]))
        self.DWho1p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
        self.DWco1p.set_value(np.float32([0.0]*self.h1_length))
        self.Dbo1p.set_value(np.float32([0.0]*self.h1_length))
        #state:
        self.DWxc1p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h1_length)]))
        self.DWhc1p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
        self.Dbc1p.set_value(np.float32([0.0]*self.h1_length))
        ###### second layer RMSprop previous gradients ###### 
        #input gate:
        self.DWxi2p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h2_length)]))
        self.DWxj2p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
        self.DWhi2p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]))
        self.DWci2p.set_value(np.float32([0.0]*self.h2_length))
        self.Dbi2p.set_value(np.float32([0.0]*self.h2_length))
        #forget gate:
        self.DWxf2p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h2_length)]))
        self.DWfj2p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
        self.DWhf2p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]))
        self.DWcf2p.set_value(np.float32([0.0]*self.h2_length))
        self.Dbf2p.set_value(np.float32([0.0]*self.h2_length))
        #output gate:
        self.DWxo2p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h2_length)]))
        self.DWoj2p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
        self.DWho2p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]))
        self.DWco2p.set_value(np.float32([0.0]*self.h2_length))
        self.Dbo2p.set_value(np.float32([0.0]*self.h2_length))
        #state:
        self.DWxc2p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h2_length)]))
        self.DWcj2p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
        self.DWhc2p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)]))
        self.Dbc2p.set_value(np.float32([0.0]*self.h2_length))
        ###### third layer RMSprop previous gradients ###### 
        #input gate:
        self.DWxi3p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h3_length)]))
        self.DWxj3p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]))
        self.DWhi3p.set_value(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]))
        self.DWci3p.set_value(np.float32([0.0]*self.h3_length))
        self.Dbi3p.set_value(np.float32([0.0]*self.h3_length))
        #forget gate:
        self.DWxf3p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h3_length)]))
        self.DWfj3p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]))
        self.DWhf3p.set_value(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]))
        self.DWcf3p.set_value(np.float32([0.0]*self.h3_length))
        self.Dbf3p.set_value(np.float32([0.0]*self.h3_length))
        #output gate:
        self.DWxo3p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h3_length)]))
        self.DWoj3p.set_value(np.float32([[0.0 for m in range(self.h2_length)] for n in range(self.h3_length)]))
        self.DWho3p.set_value(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]))
        self.DWco3p.set_value(np.float32([0.0]*self.h3_length))
        self.Dbo3p.set_value(np.float32([0.0]*self.h3_length))
        #state:
        self.DWxc3p.set_value(np.float32([[0.0]*self.io_length for n in range(self.h3_length)]))
        self.DWcj3p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]))
        self.DWhc3p.set_value(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)]))
        self.Dbc3p.set_value(np.float32([0.0]*self.h3_length))
        ###### output layer RMSprop previous gradients ###### 
        self.DWhy1p.set_value(np.float32([[0.0]*self.h1_length for n in range(self.io_length)]))
        self.DWhy2p.set_value(np.float32([[0.0]*self.h2_length for n in range(self.io_length)]))
        self.DWhy3p.set_value(np.float32([[0.0]*self.h3_length for n in range(self.io_length)]))
        self.Dbyp.set_value(np.float32([0.0]*self.io_length))
        
    def forwardPassGen(self, in1, htd1_1, htd1_2, htd1_3, ctd1_1, ctd1_2, ctd1_3):
        '''
        '''       

        [gyt, gYt, ght_1, ght_2, ght_3, gct_1, gct_2, gct_3, 
                git_1, git_2, git_3, gft_1, gft_2, gft_3, got_1, got_2, got_3,
                gIt_1, gIt_2, gIt_3, gFt_1, gFt_2, gFt_3, gOt_1, gOt_2, gOt_3, gCt_1, gCt_2, gCt_3] = self.forwardPass(in1, htd1_1, htd1_2, htd1_3, ctd1_1, ctd1_2, ctd1_3)
            
        outProb = T.switch(gyt > np.float32(0.05), gyt, np.float32(0.0)) #sigmoid output itself is probability, no need to normalise
        fdbkSamples = self.T_rng.binomial(size=(1,self.io_length), p=outProb, dtype=theano.config.floatX)
 
        return [fdbkSamples[0], gYt, ght_1, ght_2, ght_3, gct_1, gct_2, gct_3, 
                git_1, git_2, git_3, gft_1, gft_2, gft_3, got_1, got_2, got_3,
                gIt_1, gIt_2, gIt_3, gFt_1, gFt_2, gFt_3, gOt_1, gOt_2, gOt_3, gCt_1, gCt_2, gCt_3, outProb]

       
    
        
 
    def forwardPass(self, in1, htd1_1, htd1_2, htd1_3, ctd1_1, ctd1_2, ctd1_3): #for use in scan(), arguments needs to be ordered as: "sequences", "input_info", "non-sequences"
        '''
        One step forward in time: given memory states and hidden layer outputs at time (t-1), compute memory states, outputs of each layer, 
            and internal gate variables at time t

    
        arguments:

            htd1_1 (theano tensor.vector) : h1(t-1), output of (hidden) layer 1 delayed by one time step
            htd1_2 (theano tensor.vector) : h2(t-1), output of (hidden) layer 2 delayed by one time step
            htd1_3 (theano tensor.vector) : h3(t-1), output of (hidden) layer 3 delayed by one time step

            ctd1_1 (theano tensor.vector) : c1(t-1), layer 1 memory state delayed by one time step
            ctd1_2 (theano tensor.vector) : c2(t-1), layer 2 memory state delayed by one time step
            ctd1_3 (theano tensor.vector) : c3(t-1), layer 3 memory state delayed by one time step

            outputs (modifies instance variables):

            self.yt, self.Yt (theano tensor.vector) : network output, network output beforen nonlinearity
            self.ht_1, self.ht_2, self.ht_3 (theano tensor.vector): output of each layer 
            self.ct_1, self.ct_2, self.ct_3 (theano tensor.vector): memory state of each layer

            self.it_1, self.it_2, self.it_3 (theano tensor.vector): output of input gates of each layer 
            self.ft_1, self.ft_2, self.ft_3 (theano tensor.vector): output of forget gates of each layer 
            self.ot_1, self.ot_2, self.ot_3 (theano tensor.vector): output of output gates of each layer 

            self.It_1, self.It_2, self.It_3 (theano tensor.vector): output of input gates of each layer before nonlinearity
            self.Ft_1, self.Ft_2, self.Ft_3 (theano tensor.vector): output of forget gates of each layer before nonlinearity
            self.Ot_1, self.Ot_2, self.Ot_3 (theano tensor.vector): output of output gates of each layer before nonlinearity

            self.Ct_1, self.Ct_2, self.Ct_3 (theano tensor.vector): memory state of each layer before nonlinearity

        
        '''

        
        ###### layer 1 matrixes: ###### 
        #input gate
        self.It_1 = T.dot(self.Wxi_1,in1) + T.dot(self.Whi_1, htd1_1) + T.mul(self.Wci_1, ctd1_1) + self.bi_1;  self.it_1 = T.nnet.sigmoid(self.It_1)
        #forget gate
        self.Ft_1 = T.dot(self.Wxf_1, in1) + T.dot(self.Whf_1, htd1_1) + T.mul(self.Wcf_1, ctd1_1) + self.bf_1;  self.ft_1 = T.nnet.sigmoid(self.Ft_1)
        #state
        self.Ct_1 = T.dot(self.Wxc_1, in1) + T.dot(self.Whc_1, htd1_1) + self.bc_1
        self.ct_1 = T.mul(self.ft_1, ctd1_1) + T.mul(self.it_1, self.tanh(self.Ct_1)) #T.mul(self.It_1, T.mul(np.float(2.0),T.nnet.sigmoid(self.Ct_1) - np.float32(1.0)))
        #output gate
        self.Ot_1 = T.dot(self.Wxo_1, in1) + T.dot(self.Who_1, htd1_1) + T.mul(self.Wco_1, self.Ct_1) + self.bo_1;  self.ot_1 = T.nnet.sigmoid(self.Ot_1)
        #outputs
        self.ht_1 = T.mul(self.ot_1, self.tanh(self.ct_1))#T.mul(np.float(2.0),T.nnet.sigmoid(self.Ct_1) - np.float32(1.0)))
        
        ###### layer 2 matrixes: ###### 
        #input gate
        self.It_2 = T.dot(self.Wxi_2,in1) + T.dot(self.Wxj_2, self.ht_1) + T.dot(self.Whi_2, htd1_2) + T.mul(self.Wci_2, ctd1_2) + self.bi_2;  self.it_2 = T.nnet.sigmoid(self.It_2)
        #forget gate
        self.Ft_2 = T.dot(self.Wxf_2, in1) + T.dot(self.Wfj_2, self.ht_1) + T.dot(self.Whf_2, htd1_2) + T.mul(self.Wcf_2, ctd1_2) + self.bf_2;  self.ft_2 = T.nnet.sigmoid(self.Ft_2)
        #state
        self.Ct_2 = T.dot(self.Wxc_2, in1) + T.dot(self.Wcj_2, self.ht_1) + T.dot(self.Whc_2, htd1_2) + self.bc_2
        self.ct_2 = T.mul(self.ft_2, ctd1_2) + T.mul(self.it_2, self.tanh(self.Ct_2))#T.mul(np.float(2.0),T.nnet.sigmoid(self.Ct_2) - np.float32(1.0)))
        #output gate
        self.Ot_2 = T.dot(self.Wxo_2, in1) + T.dot(self.Woj_2, self.ht_1) + T.dot(self.Who_2, htd1_2) + T.mul(self.Wco_2, self.ct_2) + self.bo_2;  self.ot_2 = T.nnet.sigmoid(self.Ot_2)
        #outputs
        self.ht_2 = T.mul(self.ot_2, self.tanh(self.ct_2)) #T.mul(np.float(2.0),T.nnet.sigmoid(self.ct_2) - np.float32(1.0)))       
         
        ###### layer 3 matrixes: ###### 
        #input gate
        self.It_3 = T.dot(self.Wxi_3,in1) + T.dot(self.Wxj_3, self.ht_2) + T.dot(self.Whi_3, htd1_3) + T.mul(self.Wci_3, ctd1_3) + self.bi_3;  self.it_3 = T.nnet.sigmoid(self.It_3)
        #forget gate
        self.Ft_3 = T.dot(self.Wxf_3, in1) + T.dot(self.Wfj_3, self.ht_2) + T.dot(self.Whf_3, htd1_3) + T.mul(self.Wcf_3, ctd1_3) + self.bf_3;  self.ft_3 = T.nnet.sigmoid(self.Ft_3)
        #state
        self.Ct_3 = T.dot(self.Wxc_3, in1) + T.dot(self.Wcj_3, self.ht_2) + T.dot(self.Whc_3, htd1_3) + self.bc_3
        self.ct_3 = T.mul(self.ft_3, ctd1_3) + T.mul(self.it_3, self.tanh(self.Ct_3)) #T.mul(np.float(2.0),T.nnet.sigmoid(self.Ct_3) - np.float32(1.0)))
        #output gate
        self.Ot_3 = T.dot(self.Wxo_3, in1) + T.dot(self.Woj_3, self.ht_2) + T.dot(self.Who_3, htd1_3) + T.mul(self.Wco_3, self.ct_3) + self.bo_3;  self.ot_3 = T.nnet.sigmoid(self.Ot_3)
        #outputs
        self.ht_3 = T.mul(self.ot_3, self.tanh(self.ct_3)) #T.mul(np.float(2.0),T.nnet.sigmoid(self.ct_3) - np.float32(1.0)))       
            
        ##### output layer matrix #####
        self.Yt = T.dot(self.Why_1,self.ht_1) + T.dot(self.Why_2,self.ht_2) + T.dot(self.Why_3, self.ht_3) + self.by 
        self.yt = T.nnet.sigmoid(self.Yt)    
        
        #likely problems with this... Ct, ct mixed...
        #return [self.yt, self.Yt, self.ht_1, self.ht_2, self.ht_3, self.Ct_1, self.ct_2, self.ct_3, 
        #        self.It_1, self.it_2, self.it_3, self.Ft_1, self.ft_2, self.ft_3, self.Ot_1, self.ot_2, self.ot_3,
        #        self.It_1, self.It_2, self.It_3, self.Ft_1, self.Ft_2, self.Ft_3, self.Ot_1, self.Ot_2, self.Ot_3, self.Ct_1, self.Ct_2, self.Ct_3]

        return [self.yt, self.Yt, self.ht_1, self.ht_2, self.ht_3, self.ct_1, self.ct_2, self.ct_3, 
                self.it_1, self.it_2, self.it_3, self.ft_1, self.ft_2, self.ft_3, self.ot_1, self.ot_2, self.ot_3,
                self.It_1, self.It_2, self.It_3, self.Ft_1, self.Ft_2, self.Ft_3, self.Ot_1, self.Ot_2, self.Ot_3, self.Ct_1, self.Ct_2, self.Ct_3]
    
    
    
    def backwardPass(self, yt, Yt, kt, 
                     Itp1_3, Itp1_2, Itp1_1, Ftp1_3, Ftp1_2, Ftp1_1, Otp1_3, Otp1_2, Otp1_1, Ctp1_3, Ctp1_2, Ctp1_1,
                     It_3, It_2, It_1, Ft_3, Ft_2, Ft_1, Ot_3, Ot_2, Ot_1, Ct_3, Ct_2, Ct_1,
                     ctd1_3, ctd1_2, ctd1_1, 
                     ftp1_3, ftp1_2, ftp1_1, itp1_3, itp1_2, itp1_1,
                     it_3, it_2, it_1, ot_3, ot_2, ot_1, ct_3, ct_2, ct_1,
                     D_ctp1_3, D_ctp1_2, D_ctp1_1, D_otp1_3, D_otp1_2, D_otp1_1, D_ftp1_3, D_ftp1_2, D_ftp1_1, D_itp1_3, D_itp1_2, D_itp1_1): 
        '''
        back propagation computation of error signals of outputs, outputs of gates and memory states at time t given same signals and error signals at time t+1

        arguments:
            yt, Yt (theano tensor.vector): network output and network output before nonlinearity at time t

            Itp1_3, Itp1_2, Itp1_1 (theano tensor.vector): I3(t+1), I2(t+1), I1(t+1), input gate outputs across three layers before nonlinearity at t+1
            Ftp1_3, Ftp1_2, Ftp1_1 (theano tensor.vector): F3(t+1), F2(t+1), F1(t+1), forget gate outputs across three layers before nonlinearity at t+1
            Otp1_3, Otp1_2, Otp1_1 (theano tensor.vector): O3(t+1), O2(t+1), O1(t+1), output gate outputs across three layers before nonlinearity at t+1

            ctd1_3, ctd1_2, ctd1_1 (theano tensor.vector): c3(t-1), c2(t-1), c1(t-1), memory states across three layers before nonlinearity at t-1

            ftp1_3, ftp1_2, ftp1_1 (theano tensor.vector): f3(t+1), f2(t+1), f1(t+1), forget gate outputs across three layers at t+1
            itp1_3, itp1_2, itp1_1 (theano tensor.vector): i3(t+1), i2(t+1), i1(t+1), input gate outputs across three layers at t+1

            it_3, it_2, it_1 (theano tensor.vector): i3(t), i2(t), i1(t), input gate outputs across three layers at t
            ot_3, ot_2, ot_1 (theano tensor.vector): o3(t), o2(t), o1(t), output gate outputs across three layers at t

            ct_3, ct_2, ct_1 (theano tensor.vector): c3(t), c2(t), c1(t), memory states across three layers at t

            D_ctp1_3, D_ctp1_2, D_ctp1_1 (theano tensor.vector): c3'(t+1), c2'(t+1), c1'(t+1), memory states error across three layers at t+1
            D_otp1_3, D_otp1_2, D_otp1_1 (theano tensor.vector): o3'(t+1), o2'(t+1), o1'(t+1), output gate error across three layers at t+1
            D_ftp1_3, D_ftp1_2, D_ftp1_1 (theano tensor.vector): f3'(t+1), f2'(t+1), f1'(t+1), forget gate error across three layers at t+1 
            D_itp1_3, D_itp1_2, D_itp1_1 (theano tensor.vector): i3'(t+1), i2'(t+1), i1'(t+1), input gate error across three layers at t+1 
            
        returns:
            sqrCost (single element theano tensor.vector): cost function at time t
            D_Yt (theano tensor.vector): Y'(t), network output error at time t
            
            D_It_3, D_It_2, D_It_1 (theano tensor.vector): I3'(t), I2'(t), I1'(t), error of input gate signal before nonlinearity across all layers at time t
            D_Ft_3, D_Ft_2, D_Ft_1 (theano tensor.vector): F3'(t), F2'(t), F1'(t), error of forget gate signal before nonlinearity across all layers at time t
            D_Ot_3, D_Ot_2, D_Ot_1 (theano tensor.vector): O3'(t), O2'(t), O1'(t), error of output gate signal before nonlinearity across all layers at time t
            D_Ct_3, D_Ct_2, D_Ct_1 (theano tensor.vector): C3'(t), C2'(t), C1'(t), error of memory state before nonlinearity across all layers at time t
 
            D_ct_3, D_ct_2, D_ct_1 (theano tensor.vector): c3'(t), c2'(t), c1'(t), error of memory state across all layers at time t
            D_ot_3, D_ot_2, D_ot_1 (theano tensor.vector): o3'(t), o2'(t), o1'(t), error of output gate across all layers at time t
            D_ft_3, D_ft_2, D_ft_1 (theano tensor.vector): f3'(t), f2'(t), f1'(t), error of forget gate across all layers at time t
            D_it_3, D_it_2, D_it_1 (theano tensor.vector): i3'(t), i2'(t), i1'(t), error of input gate across all layers at time t

            '''

        D_yt = - (kt / T.maximum(self.epsilon, yt)) + ((np.float32(1.0) - kt) / (1-T.minimum((np.float32(1.0) - self.epsilon),yt))) #cross entropy cost function
        #D_yt = yt - kt #sqr error cost function
        D_Yt = T.mul(D_yt, self.gdot(Yt))
        
        #costEst = T.sum(T.mul(np.float32(0.5), T.mul(kt-yt,kt-yt)), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
        costEst = T.sum(-T.mul(kt,T.log(yt)) - T.mul((np.float32(1.0)-kt), T.log(np.float32(1.0)-yt)))


        ###### start with layer 3 for back prop ##### 
        D_ht_3 = T.dot(T.mul(self.gdot(Yt), D_yt), self.Why_3) + T.dot(T.mul(D_itp1_3, self.gdot(Itp1_3)), self.Whi_3) + T.dot(T.mul(D_ftp1_3, self.gdot(Ftp1_3)), self.Whf_3) \
                    + T.dot(T.mul(D_otp1_3, self.gdot(Otp1_3)), self.Who_3) + T.dot(T.mul(T.mul(D_ctp1_3, itp1_3), self.tdot(Ctp1_3)), self.Whc_3)
        D_ct_3 = T.mul(T.mul(D_itp1_3, self.gdot(Itp1_3)), self.Wci_3) + T.mul(T.mul(D_ftp1_3, self.gdot(Ftp1_3)),self.Wcf_3) \
                    + T.mul(T.mul(D_otp1_3, self.gdot(Otp1_3)), self.Wco_3) + T.mul(D_ctp1_3, ftp1_3) +  T.mul(T.mul(D_ht_3, ot_3), self.tdot(ct_3))
        D_Ct_3 = T.mul(T.mul(D_ct_3, it_3), self.tdot(Ct_3))
        D_ot_3 = T.mul(D_ht_3, self.tanh(ct_3))
        D_Ot_3 = T.mul(D_ot_3, self.gdot(Ot_3))
        D_ft_3 = T.mul(D_ct_3, ctd1_3)
        D_Ft_3 = T.mul(D_ft_3, self.gdot(Ft_3))
        D_it_3 = T.mul(D_ct_3, self.tanh(Ct_3))
        D_It_3 = T.mul(D_it_3, self.gdot(It_3))
        ###### layer 2 back prop ###### last 2 lines of D_ht_2 takes inputs from third layer
        D_ht_2 = T.dot(T.mul(self.gdot(Yt), D_yt), self.Why_2) + T.dot(T.mul(D_itp1_2, self.gdot(Itp1_2)), self.Whi_2) + T.dot(T.mul(D_ftp1_2, self.gdot(Ftp1_2)), self.Whf_2)  \
                    + T.dot(T.mul(D_otp1_2, self.gdot(Otp1_2)), self.Who_2) + T.dot(T.mul(T.mul(D_ctp1_2, itp1_2), self.tdot(Ctp1_2)), self.Whc_2) \
                    + T.dot(T.mul(D_it_3, self.gdot(It_3)), self.Wxj_3) + T.dot(T.mul(D_ft_3, self.gdot(Ft_3)), self.Wfj_3) \
                    + T.dot(T.mul(D_ot_3, self.gdot(Ot_3)),self.Woj_3) + T.dot(T.mul(T.mul(D_ct_3, it_3),self.tdot(Ct_3)), self.Wcj_3)
        D_ct_2 = T.mul(T.mul(D_itp1_2, self.gdot(Itp1_2)), self.Wci_2) + T.mul(T.mul(D_ftp1_2, self.gdot(Ftp1_2)),self.Wcf_2) \
                    + T.mul(T.mul(D_otp1_2, self.gdot(Otp1_2)), self.Wco_2) + T.mul(D_ctp1_2, ftp1_2) + T.mul(T.mul(D_ht_2, ot_2), self.tdot(ct_2))
        D_Ct_2 = T.mul(T.mul(D_ct_2, it_2), self.tdot(Ct_2))
        D_ot_2 = T.mul(D_ht_2, self.tanh(ct_2))
        D_Ot_2 = T.mul(D_ot_2, self.gdot(Ot_2))
        D_ft_2 = T.mul(D_ct_2, ctd1_2)
        D_Ft_2 = T.mul(D_ft_2,self.gdot(Ft_2))
        D_it_2 = T.mul(D_ct_2, self.tanh(Ct_2))
        D_It_2 = T.mul(D_it_2, self.gdot(It_2))
        ###### layer 1 back prop ###### last 2 lines of D_ht_1 takes inputs from second layer
        D_ht_1 = T.dot(T.mul(self.gdot(Yt), D_yt), self.Why_1) + T.dot(T.mul(D_itp1_1, self.gdot(Itp1_1)), self.Whi_1) + T.dot(T.mul(D_ftp1_1, self.gdot(Ftp1_1)), self.Whf_1) \
                    + T.dot(T.mul(D_otp1_1, self.gdot(Otp1_1)), self.Who_1) + T.dot(T.mul(T.mul(D_ctp1_1, itp1_1), self.tdot(Ctp1_1)), self.Whc_1) \
                    + T.dot(T.mul(D_it_2, self.gdot(It_2)), self.Wxj_2) + T.dot(T.mul(D_ft_2, self.gdot(Ft_2)), self.Wfj_2) \
                    + T.dot(T.mul(D_ot_2, self.gdot(Ot_2)),self.Woj_2) + T.dot(T.mul(T.mul(D_ct_2, it_2),self.tdot(Ct_2)), self.Wcj_2)
        D_ct_1 = T.mul(T.mul(D_itp1_1, self.gdot(Itp1_1)), self.Wci_1) + T.mul(T.mul(D_ftp1_1, self.gdot(Ftp1_1)),self.Wcf_1) \
                    + T.mul(T.mul(D_otp1_1, self.gdot(Otp1_1)), self.Wco_1) + T.mul(D_ctp1_1, ftp1_1) + T.mul(T.mul(D_ht_1, ot_1), self.tdot(ct_1))
        D_Ct_1 = T.mul(T.mul(D_ct_1, it_1), self.tdot(Ct_1))
        D_ot_1 = T.mul(D_ht_1, self.tanh(ct_1))
        D_Ot_1 = T.mul(D_ot_1, self.gdot(Ot_1))
        D_ft_1 = T.mul(D_ct_1, ctd1_1)
        D_Ft_1 = T.mul(D_ft_1,self.gdot(Ft_1))
        D_it_1 = T.mul(D_ct_1, self.tanh(Ct_1))
        D_It_1 = T.mul(D_it_1, self.gdot(It_1))       
               
        return [costEst, D_Yt, D_It_3, D_It_2, D_It_1, D_Ft_3, D_Ft_2, D_Ft_1, D_Ct_3, D_Ct_2, D_Ct_1, D_Ot_3, D_Ot_2, D_Ot_1, 
                    D_ct_3, D_ct_2, D_ct_1, D_ot_3, D_ot_2, D_ot_1, D_ft_3, D_ft_2, D_ft_1, D_it_3, D_it_2, D_it_1] # first line is for weights update, second line is to be feedback
        
    def gClip(self, inTensor):
        '''
        upper clips magnitude of input tensor (theano symbolic) to self.gradClip

        argument:
        inTensor (theano tensor): input tensor be either matrix or vector	

        return:
            sign(input) * min(|input|, self.gradClip)

        '''
        return T.mul(T.sgn(inTensor), T.minimum(self.gradClip,T.abs_(inTensor)))
    
        
    def tanh(self, inVec):
        '''
        computes tanh(input), input is theano tensor (symbolic)
        '''
        return T.mul(np.float32(2.0), T.nnet.sigmoid(T.mul(np.float32(2.0), inVec))) - np.float32(1.0)
        
    def RMSgrad(self, prevGrad, newGrad):
        gradSqr = T.mul(np.float32(0.9), T.mul(prevGrad, prevGrad)) + T.mul(np.float32(0.1), T.mul(newGrad, newGrad))
        return (newGrad / T.sqrt(T.maximum(self.epsilon, gradSqr)))
    
    def gdot(self, inVec):
        ''' gdot(input) : deriviative of sigmoid '''
        return T.mul(T.nnet.sigmoid(inVec), np.float32(1.0) - T.nnet.sigmoid(inVec))
    
    
    def tdot(self, inVec):
        ''' 
        computes derivative of tanh: tanh'(input), input is theano tensor (symbolic)
        derivation starts by writing tanh as 2S(x) - 1, S(x) as sigmoid function
        '''
        return T.mul(np.float32(4.0), self.gdot(T.mul(np.float32(2.0),inVec)))# - np.float32(1.0)
    
    def d1(self,inMat):
        '''
        attaches zero vector in front of matrixc and throw away last row
        created to shorten syntax
        '''
        return T.concatenate([[T.zeros_like(inMat[0], dtype=theano.config.floatX)], inMat[0:inMat.shape[0]-1]], axis=0)
        
    def p1(self,inMat):
        '''
        attaches zero vector at end of matrixc and throw away first row
        created to shorten syntax
        '''
        return T.concatenate([inMat[1:inMat.shape[0]], [T.zeros_like(inMat[0], dtype=theano.config.floatX)]], axis=0)
    
                                    
    def loadParameters(self, fileName):
        '''
        loads saved parameters from npz file, once loaded training etc can resume

        argument:
        fileName (string): file name of saved parameter file without npz file extension
        '''

        #print("before loading, omega = " + str(self.T_omega.eval()))
        #with open(fileName+'.npz', "rb") as file_:
        #    loadedFile = np.load(file_)
        lF = np.load(fileName + '.npz')
        self.instantaneousCost=lF['instantaneousCost']; self.gradClip=lF['gradClip']; self.h1_length=int(lF['h1_length']); self.h2_length=int(lF['h2_length']); self.io_length=int(lF['io_length']) 
        print("loaded (h1_length,h2_length,io_length) = (" + str(self.h1_length) + "," + str(self.h2_length) + "," + str(self.io_length) + ")")
        self.R1 = np.float32(lF['R1']); self.R2 = np.float32(lF['R2']); self.R3 = np.float32(lF['R3'])
        self.Wxi_1.set_value(lF['Wxi_1']); self.Wxi_2.set_value(lF['Wxi_2']); self.Wxi_3.set_value(lF['Wxi_3'])
        self.Wxf_1.set_value(lF['Wxf_1']); self.Wxf_2.set_value(lF['Wxf_2']); self.Wxf_3.set_value(lF['Wxf_3'])
        self.Wxo_1.set_value(lF['Wxo_1']); self.Wxo_2.set_value(lF['Wxo_2']); self.Wxo_3.set_value(lF['Wxo_3'])
        self.Wxc_1.set_value(lF['Wxc_1']); self.Wxc_2.set_value(lF['Wxc_2']); self.Wxc_3.set_value(lF['Wxc_3'])
        self.Whi_1.set_value(lF['Whi_1']); self.Whi_2.set_value(lF['Whi_2']); self.Whi_3.set_value(lF['Whi_3'])
        self.Whf_1.set_value(lF['Whf_1']); self.Whf_2.set_value(lF['Whf_2']); self.Whf_3.set_value(lF['Whf_3'])
        self.Who_1.set_value(lF['Who_1']); self.Who_2.set_value(lF['Who_2']); self.Who_3.set_value(lF['Who_3'])
        self.Whc_1.set_value(lF['Whc_1']); self.Whc_2.set_value(lF['Whc_2']); self.Whc_3.set_value(lF['Whc_3'])
        self.Why_1.set_value(lF['Why_1']); self.Why_2.set_value(lF['Why_2']); self.Why_3.set_value(lF['Why_3'])
        self.Wxj_2.set_value(lF['Wxj_2']); self.Wxj_3.set_value(lF['Wxj_3']);
        self.Wfj_2.set_value(lF['Wfj_2']); self.Wfj_3.set_value(lF['Wfj_3']);
        self.Wcj_2.set_value(lF['Wcj_2']); self.Wcj_3.set_value(lF['Wcj_3']);
        self.Woj_2.set_value(lF['Woj_2']); self.Woj_3.set_value(lF['Woj_3']);
        self.Wci_1.set_value(lF['Wci_1']); self.Wci_2.set_value(lF['Wci_2']); self.Wci_3.set_value(lF['Wci_3']);
        self.Wcf_1.set_value(lF['Wcf_1']); self.Wcf_2.set_value(lF['Wcf_2']); self.Wcf_3.set_value(lF['Wcf_3']);
        self.Wco_1.set_value(lF['Wco_1']); self.Wco_2.set_value(lF['Wco_2']); self.Wco_3.set_value(lF['Wco_3']);
        self.loss.set_value(lF['loss'])
        self.bi_1.set_value(lF['bi_1']); self.bi_2.set_value(lF['bi_2']); self.bi_3.set_value(lF['bi_3'])
        self.bf_1.set_value(lF['bf_1']); self.bf_2.set_value(lF['bf_2']); self.bf_3.set_value(lF['bf_3'])
        self.bc_1.set_value(lF['bc_1']); self.bc_2.set_value(lF['bc_2']); self.bc_3.set_value(lF['bc_3'])
        self.bo_1.set_value(lF['bo_1']); self.bo_2.set_value(lF['bo_2']); self.bo_3.set_value(lF['bo_3'])
        self.by.set_value(lF['by'])
        self.DWxi1p.set_value(lF['DWxi1p']); self.DWxi2p.set_value(lF['DWxi2p']); self.DWxi3p.set_value(lF['DWxi3p'])
        self.DWxf1p.set_value(lF['DWxf1p']); self.DWxf2p.set_value(lF['DWxf2p']); self.DWxf3p.set_value(lF['DWxf3p'])
        self.DWxo1p.set_value(lF['DWxo1p']); self.DWxo2p.set_value(lF['DWxo2p']); self.DWxo3p.set_value(lF['DWxo3p']) 
        self.DWxc1p.set_value(lF['DWxc1p']); self.DWxc2p.set_value(lF['DWxc2p']); self.DWxc3p.set_value(lF['DWxc3p'])
        self.DWhi1p.set_value(lF['DWhi1p']); self.DWhi2p.set_value(lF['DWhi2p']); self.DWhi3p.set_value(lF['DWhi3p'])
        self.DWhf1p.set_value(lF['DWhf1p']); self.DWhf2p.set_value(lF['DWhf2p']); self.DWhf3p.set_value(lF['DWhf3p'])
        self.DWho1p.set_value(lF['DWho1p']); self.DWho2p.set_value(lF['DWho2p']); self.DWho3p.set_value(lF['DWho3p'])
        self.DWhc1p.set_value(lF['DWhc1p']); self.DWhc2p.set_value(lF['DWhc2p']); self.DWhc3p.set_value(lF['DWhc3p'])
        self.DWhy1p.set_value(lF['DWhy1p']); self.DWhy2p.set_value(lF['DWhy2p']); self.DWhy3p.set_value(lF['DWhy3p']) 
        self.DWxj2p.set_value(lF['DWxj2p']); self.DWxj3p.set_value(lF['DWxj3p']);
        self.DWfj2p.set_value(lF['DWfj2p']); self.DWfj3p.set_value(lF['DWfj3p']); 
        self.DWcj2p.set_value(lF['DWcj2p']); self.DWcj3p.set_value(lF['DWcj3p']);
        self.DWoj2p.set_value(lF['DWoj2p']); self.DWoj3p.set_value(lF['DWoj3p']);
        self.DWci1p.set_value(lF['DWci1p']); self.DWci2p.set_value(lF['DWci2p']); self.DWci3p.set_value(lF['DWci3p'])
        self.DWcf1p.set_value(lF['DWcf1p']); self.DWcf2p.set_value(lF['DWcf2p']); self.DWcf3p.set_value(lF['DWcf3p'])
        self.DWco1p.set_value(lF['DWco1p']); self.DWco2p.set_value(lF['DWco2p']); self.DWco3p.set_value(lF['DWco3p']) 
        self.Dbi1p.set_value(lF['Dbi1p']); self.Dbi2p.set_value(lF['Dbi2p']); self.Dbi3p.set_value(lF['Dbi3p'])
        self.Dbf1p.set_value(lF['Dbf1p']); self.Dbf2p.set_value(lF['Dbf2p']); self.Dbf3p.set_value(lF['Dbf3p'])
        self.Dbc1p.set_value(lF['Dbc1p']); self.Dbc2p.set_value(lF['Dbc2p']); self.Dbc3p.set_value(lF['Dbc3p'])
        self.Dbo1p.set_value(lF['Dbo1p']); self.Dbo2p.set_value(lF['Dbo2p']); self.Dbo3p.set_value(lF['Dbo3p'])
        self.Dbyp.set_value(lF['Dbyp'])        
       
        #print("after loading, omega = " + str(self.T_omega.eval()))   
        self.parameterLoaded = True  
                                  
    
        
    def saveParameters(self, fileName):
        '''
        saves network status to npz file, once saved, can use self.loadParameters(self, fileName) to load state and resume training etc

        argument:
        fileName (string): npz filename to be saved, overwrites if exists. supply filename without .npz extension.
        '''
        #print("before saving, omega = " + str(self.T_omega.eval()))
        np.savez(fileName, instantaneousCost = self.instantaneousCost, gradClip = self.gradClip, h1_length = self.h1_length, h2_length = self.h2_length, io_length = self.io_length, 
                                    R1 = np.float32(self.R1), R2 = np.float32(self.R2), R3 = np.float32(self.R3),
                                    Wxi_1 = self.Wxi_1.eval(), Wxi_2 = self.Wxi_2.eval(), Wxi_3 = self.Wxi_3.eval(),
                                    Wxf_1 = self.Wxf_1.eval(), Wxf_2 = self.Wxf_2.eval(), Wxf_3 = self.Wxf_3.eval(),
                                    Wxo_1 = self.Wxo_1.eval(), Wxo_2 = self.Wxo_2.eval(), Wxo_3 = self.Wxo_3.eval(),
                                    Wxc_1 = self.Wxc_1.eval(), Wxc_2 = self.Wxc_2.eval(), Wxc_3 = self.Wxc_3.eval(),
                                    Whi_1 = self.Whi_1.eval(), Whi_2 = self.Whi_2.eval(), Whi_3 = self.Whi_3.eval(),
                                    Whf_1 = self.Whf_1.eval(), Whf_2 = self.Whf_2.eval(), Whf_3 = self.Whf_3.eval(),
                                    Who_1 = self.Who_1.eval(), Who_2 = self.Who_2.eval(), Who_3 = self.Who_3.eval(),
                                    Whc_1 = self.Whc_1.eval(), Whc_2 = self.Whc_2.eval(), Whc_3 = self.Whc_3.eval(),
                                    Why_1 = self.Why_1.eval(), Why_2 = self.Why_2.eval(), Why_3 = self.Why_3.eval(),
                                    Wxj_2 = self.Wxj_2.eval(), Wxj_3 = self.Wxj_3.eval(),
                                    Wfj_2 = self.Wfj_2.eval(), Wfj_3 = self.Wfj_3.eval(),
                                    Wcj_2 = self.Wcj_2.eval(), Wcj_3 = self.Wcj_3.eval(),
                                    Woj_2 = self.Woj_2.eval(), Woj_3 = self.Woj_3.eval(),
                                    Wci_1 = self.Wci_1.eval(), Wci_2 = self.Wci_2.eval(), Wci_3 = self.Wci_3.eval(),
                                    Wcf_1 = self.Wcf_1.eval(), Wcf_2 = self.Wcf_2.eval(), Wcf_3 = self.Wcf_3.eval(),
                                    Wco_1 = self.Wco_1.eval(), Wco_2 = self.Wco_2.eval(), Wco_3 = self.Wco_3.eval(),
                                    loss = self.loss.eval(),
                                    bi_1 = self.bi_1.eval(), bi_2 = self.bi_2.eval(), bi_3 = self.bi_3.eval(),
                                    bf_1 = self.bf_1.eval(), bf_2 = self.bf_2.eval(), bf_3 = self.bf_3.eval(),
                                    bc_1 = self.bc_1.eval(), bc_2 = self.bc_2.eval(), bc_3 = self.bc_3.eval(),
                                    bo_1 = self.bo_1.eval(), bo_2 = self.bo_2.eval(), bo_3 = self.bo_3.eval(),
                                    by = self.by.eval(),
                                    DWxi1p = self.DWxi1p.eval(), DWxi2p = self.DWxi2p.eval(), DWxi3p = self.DWxi3p.eval(),
                                    DWxf1p = self.DWxf1p.eval(), DWxf2p = self.DWxf2p.eval(), DWxf3p = self.DWxf3p.eval(),
                                    DWxo1p = self.DWxo1p.eval(), DWxo2p = self.DWxo2p.eval(), DWxo3p = self.DWxo3p.eval(), 
                                    DWxc1p = self.DWxc1p.eval(), DWxc2p = self.DWxc2p.eval(), DWxc3p = self.DWxc3p.eval(),
                                    DWhi1p = self.DWhi1p.eval(), DWhi2p = self.DWhi2p.eval(), DWhi3p = self.DWhi3p.eval(),
                                    DWhf1p = self.DWhf1p.eval(), DWhf2p = self.DWhf2p.eval(), DWhf3p = self.DWhf3p.eval(),
                                    DWho1p = self.DWho1p.eval(), DWho2p = self.DWho2p.eval(), DWho3p = self.DWho3p.eval(),
                                    DWhc1p = self.DWhc1p.eval(), DWhc2p = self.DWhc2p.eval(), DWhc3p = self.DWhc3p.eval(),
                                    DWhy1p = self.DWhy1p.eval(), DWhy2p = self.DWhy2p.eval(), DWhy3p = self.DWhy3p.eval(), 
                                    DWxj2p = self.DWxj2p.eval(), DWxj3p = self.DWxj3p.eval(),
                                    DWfj2p = self.DWfj2p.eval(), DWfj3p = self.DWfj3p.eval(), 
                                    DWcj2p = self.DWcj2p.eval(), DWcj3p = self.DWcj3p.eval(),
                                    DWoj2p = self.DWoj2p.eval(), DWoj3p = self.DWoj3p.eval(),
                                    DWci1p = self.DWci1p.eval(), DWci2p = self.DWci2p.eval(), DWci3p = self.DWci3p.eval(),
                                    DWcf1p = self.DWcf1p.eval(), DWcf2p = self.DWcf2p.eval(), DWcf3p = self.DWcf3p.eval(),
                                    DWco1p = self.DWco1p.eval(), DWco2p = self.DWco2p.eval(), DWco3p = self.DWco3p.eval(), 
                                    Dbi1p = self.Dbi1p.eval(), Dbi2p = self.Dbi2p.eval(), Dbi3p = self.Dbi3p.eval(),
                                    Dbf1p = self.Dbf1p.eval(), Dbf2p = self.Dbf2p.eval(), Dbf3p = self.Dbf3p.eval(),
                                    Dbc1p = self.Dbc1p.eval(), Dbc2p = self.Dbc2p.eval(), Dbc3p = self.Dbc3p.eval(),
                                    Dbo1p = self.Dbo1p.eval(), Dbo2p = self.Dbo2p.eval(), Dbo3p = self.Dbo3p.eval(),
                                    Dbyp = self.Dbyp.eval())
                    
                    #Wrnnout = self.Wrnnout.eval())
                 
        #print("parameters saved in: " + str(fileName) + ".npz")
        self.parameterSaved = True

    
    
    
    def gradSum(self, inRow, h1td1Row, h2td1Row, h3td1Row, D_It_1Row, D_It_2Row, D_It_3Row, D_Ft_1Row, D_Ft_2Row, D_Ft_3Row, D_Ot_1Row, D_Ot_2Row, D_Ot_3Row, D_Ct_1Row, D_Ct_2Row, D_Ct_3Row,
                D_YtRow, h1Row, h2Row, h3Row, c1td1Row, c2td1Row, c3td1Row, c1Row, c2Row, c3Row,
                Wxi1sum, Wxi2sum, Wxi3sum, Wxf1sum, Wxf2sum, Wxf3sum, Wxo1sum, Wxo2sum, Wxo3sum, Wxc1sum, Wxc2sum, Wxc3sum,
                Whi1sum, Whi2sum, Whi3sum, Whf1sum, Whf2sum, Whf3sum, Who1sum, Who2sum, Who3sum, Whc1sum, Whc2sum, Whc3sum,
                Why1sum, Why2sum, Why3sum,
                Wxj2sum, Wxj3sum, Wfj2sum, Wfj3sum, Wcj2sum, Wcj3sum, Woj2sum, Woj3sum,
                Wci1sum, Wci2sum, Wci3sum, Wcf1sum, Wcf2sum, Wcf3sum, Wco1sum, Wco2sum, Wco3sum):
     
        '''
        calculates accumulated gradient updates of weight parameters of the network, given previously accumulated gradient and error signals. 

        arguments:
            inRow (theano tensor.vector): network input at time t
            h1td1Row, h2td1Row, h3td1Row (theano tensor.vector): h1(t-1), h2(t-1), h3(t-1)
            D_It_1Row, D_It_2Row, D_It_3Row (theano tensor.vector): I1'(t), I2'(t), I3'(t)  
            D_Ft_1Row, D_Ft_2Row, D_Ft_3Row (theano tensor.vector): F1'(t), F2'(t), F3'(t)  
            D_Ot_1Row, D_Ot_2Row, D_Ot_3Row (theano tensor.vector): O1'(t), O2'(t), O3'(t)  
            D_Ct_1Row, D_Ct_2Row, D_Ct_3Row (theano tensor.vector): C1'(t), C2'(t), C3'(t)
            D_YtRow (): Y'(t)
            h1Row, h2Row, h3Row (): h1(t), h2(t), h3(t) 
            c1td1Row, c2td1Row, c3td1Row (): c1(t-1), c2(t-1), c3(t-1)
            
            c1Row, c2Row, c3Row (): c1(t), c2(t), c3(t)
                
            Wxi1sum, Wxi2sum, Wxi3sum, Wxf1sum, Wxf2sum, Wxf3sum, Wxo1sum, Wxo2sum, Wxo3sum, Wxc1sum, Wxc2sum, Wxc3sum, Whi1sum, Whi2sum, Whi3sum, 
                Whf1sum, Whf2sum, Whf3sum, Who1sum, Who2sum, Who3sum, Whc1sum, Whc2sum, Whc3sum, Why1sum, Why2sum, Why3sum, Wxj2sum, Wxj3sum, Wfj2sum, 
                Wfj3sum, Wcj2sum, Wcj3sum, Woj2sum, Woj3sum, Wci1sum, Wci2sum, Wci3sum, Wcf1sum, Wcf2sum, Wcf3sum, Wco1sum, Wco2sum, Wco3sum 
                    (theano tensor.matrix): Accumulated gradient of weights.

        returns:
            Wxi1tmp, Wxi2tmp, Wxi3tmp, Wxf1tmp, Wxf2tmp, Wxf3tmp, Wxo1tmp, Wxo2tmp, Wxo3tmp, Wxc1tmp, Wxc2tmp, Wxc3tmp,
            Whi1tmp, Whi2tmp, Whi3tmp, Whf1tmp, Whf2tmp, Whf3tmp, Who1tmp, Who2tmp, Who3tmp, Whc1tmp, Whc2tmp, Whc3tmp,
            Why1tmp, Why2tmp, Why3tmp,
            Wxj2tmp, Wxj3tmp, Wfj2tmp, Wfj3tmp, Wcj2tmp, Wcj3tmp, Woj2tmp, Woj3tmp,
            Wci1tmp, Wci2tmp, Wci3tmp, Wcf1tmp, Wcf2tmp, Wcf3tmp, Wco1tmp, Wco2tmp, Wco3tmp (theano tensor.matrix): New accumulated gradient of weights
 


        '''
        Wxi1tmp = Wxi1sum + T.outer(D_It_1Row, inRow);  Wxi2tmp = Wxi2sum + T.outer(D_It_2Row, inRow);  Wxi3tmp = Wxi3sum + T.outer(D_It_3Row, inRow)
        Wxf1tmp = Wxf1sum + T.outer(D_Ft_1Row, inRow);  Wxf2tmp = Wxf2sum + T.outer(D_Ft_2Row, inRow);  Wxf3tmp = Wxf3sum + T.outer(D_Ft_3Row, inRow)
        Wxo1tmp = Wxo1sum + T.outer(D_Ot_1Row, inRow);  Wxo2tmp = Wxo2sum + T.outer(D_Ot_2Row, inRow);  Wxo3tmp = Wxo3sum + T.outer(D_Ot_3Row, inRow)
        Wxc1tmp = Wxc1sum + T.outer(D_Ct_1Row, inRow);  Wxc2tmp = Wxc2sum + T.outer(D_Ct_2Row, inRow);  Wxc3tmp = Wxc3sum + T.outer(D_Ct_3Row, inRow)
        
        Whi1tmp = Whi1sum + T.outer(D_It_1Row, h1td1Row) 
        
        Whi2tmp = Whi2sum + T.outer(D_It_2Row, h2td1Row);  Whi3tmp = Whi3sum + T.outer(D_It_3Row, h3td1Row)
        Whf1tmp = Whf1sum + T.outer(D_Ft_1Row, h1td1Row);  Whf2tmp = Whf2sum + T.outer(D_Ft_2Row, h2td1Row);  Whf3tmp = Whf3sum + T.outer(D_Ft_3Row, h3td1Row)
        Who1tmp = Who1sum + T.outer(D_Ot_1Row, h1td1Row);  Who2tmp = Who2sum + T.outer(D_Ot_2Row, h2td1Row);  Who3tmp = Who3sum + T.outer(D_Ot_3Row, h3td1Row)
        Whc1tmp = Whc1sum + T.outer(D_Ct_1Row, h1td1Row);  Whc2tmp = Whc2sum + T.outer(D_Ct_2Row, h2td1Row);  Whc3tmp = Whc3sum + T.outer(D_Ct_3Row, h3td1Row)
        
        Why1tmp = Why1sum + T.outer(D_YtRow, h1Row);  Why2tmp = Why2sum + T.outer(D_YtRow, h2Row);  Why3tmp = Why3sum + T.outer(D_YtRow, h3Row); 
        
        Wxj2tmp = Wxj2sum + T.outer(D_It_2Row, h1Row);  Wxj3tmp = Wxj3sum + T.outer(D_It_3Row, h2Row)
        Wfj2tmp = Wfj2sum + T.outer(D_Ft_2Row, h1Row);  Wfj3tmp = Wfj3sum + T.outer(D_Ft_3Row, h2Row)
        Wcj2tmp = Wcj2sum + T.outer(D_Ct_2Row, h1Row);  Wcj3tmp = Wcj3sum + T.outer(D_Ct_3Row, h2Row)
        Woj2tmp = Woj2sum + T.outer(D_Ot_2Row, h1Row);  Woj3tmp = Woj3sum + T.outer(D_Ot_3Row, h2Row)
        
        Wci1tmp = Wci1sum + T.mul(D_It_1Row, c1td1Row);  Wci2tmp = Wci2sum + T.mul(D_It_2Row, c2td1Row);  Wci3tmp = Wci3sum + T.mul(D_It_3Row, c3td1Row)
        Wcf1tmp = Wcf1sum + T.mul(D_Ft_1Row, c1td1Row);  Wcf2tmp = Wcf2sum + T.mul(D_Ft_2Row, c2td1Row);  Wcf3tmp = Wcf3sum + T.mul(D_Ft_3Row, c3td1Row)
        Wco1tmp = Wco1sum + T.mul(D_Ot_1Row, c1Row);  Wco2tmp = Wco2sum + T.mul(D_Ot_2Row, c2Row);  Wco3tmp = Wco3sum + T.mul(D_Ot_3Row, c3Row);
        
        return [Wxi1tmp, Wxi2tmp, Wxi3tmp, Wxf1tmp, Wxf2tmp, Wxf3tmp, Wxo1tmp, Wxo2tmp, Wxo3tmp, Wxc1tmp, Wxc2tmp, Wxc3tmp,
                Whi1tmp, Whi2tmp, Whi3tmp, Whf1tmp, Whf2tmp, Whf3tmp, Who1tmp, Who2tmp, Who3tmp, Whc1tmp, Whc2tmp, Whc3tmp,
                Why1tmp, Why2tmp, Why3tmp,
                Wxj2tmp, Wxj3tmp, Wfj2tmp, Wfj3tmp, Wcj2tmp, Wcj3tmp, Woj2tmp, Woj3tmp,
                Wci1tmp, Wci2tmp, Wci3tmp, Wcf1tmp, Wcf2tmp, Wcf3tmp, Wco1tmp, Wco2tmp, Wco3tmp]
        
    def resetStates(self):
        '''
        resets all memory states of the LSTM to zero
        '''
 
        self.ctd1_1.set_value(np.float32(np.random.normal(np.float32(0.0),self.sigRand,self.h1_length)))
        self.ctd1_2.set_value(np.float32(np.random.normal(np.float32(0.0),self.sigRand,self.h2_length)))
        self.ctd1_3.set_value(np.float32(np.random.normal(np.float32(0.0),self.sigRand,self.h3_length)))
        self.htd1_1.set_value(np.float32(np.random.normal(np.float32(0.0),self.sigRand,self.h1_length)))
        self.htd1_2.set_value(np.float32(np.random.normal(np.float32(0.0),self.sigRand,self.h2_length)))
        self.htd1_3.set_value(np.float32(np.random.normal(np.float32(0.0),self.sigRand,self.h3_length)))



    def gradSingleTune(self, T_egMB, T_egOutMB, h1_cont, h2_cont, h3_cont, c1_cont, c2_cont, c3_cont):


        #conveneint zero variables for output_info initial values:
        ioh1 = np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)]))
        ioh2 = np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)]))
        ioh3 = np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)]))
        h1h1 = np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)]))
        h2h2 = np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)]))
        h3h3 = np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)]))
        h1io = np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.io_length)]))
        h2io = np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.io_length)]))
        h3io = np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.io_length)]))
        h1h2 = np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)]))
        h2h3 = np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)]))
        h1v = np.float32(np.asarray([0.0]*self.h1_length))
        h2v = np.float32(np.asarray([0.0]*self.h2_length))
        h3v = np.float32(np.asarray([0.0]*self.h3_length))

        #reset states between examples within mini-batch as well, comment out to experiment:
        h1_cont = cp.deepcopy(h1v); h2_cont = cp.deepcopy(h2v); h3_cont = cp.deepcopy(h3v)
        c1_cont = cp.deepcopy(h1v); c2_cont = cp.deepcopy(h2v); c3_cont = cp.deepcopy(h3v)


        ###### forward run ######         
        #usage of scan(): sequences will fill up input arguments first, the leftover arguments will be filled by "output_infos". outputs_info should match return pattern, 
        # with ones not feeding back marked as 'None'
        [TytAcc, TYtAcc, Th1Acc, Th2Acc, Th3Acc, Tc1Acc, Tc2Acc, Tc3Acc, 
         Tit_1Acc, Tit_2Acc, Tit_3Acc, Tft_1Acc, Tft_2Acc, Tft_3Acc, Tot_1Acc, Tot_2Acc, Tot_3Acc,
         TIt_1Acc, TIt_2Acc, TIt_3Acc, TFt_1Acc, TFt_2Acc, TFt_3Acc, TOt_1Acc, TOt_2Acc, TOt_3Acc, TCt_1Acc, TCt_2Acc, TCt_3Acc], \
        _ = theano.scan(fn=self.forwardPass, sequences=[T_egMB],
                                    outputs_info=[None, None,  h1_cont, h2_cont, h3_cont, c1_cont, c2_cont, c3_cont, 
                                                  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]) 

        

        ###### backwards run: generating error signals for all variables across all layers and across time ######
        [KLAcc, D_YtAcc, D_It_3Acc, D_It_2Acc, D_It_1Acc, D_Ft_3Acc, D_Ft_2Acc, D_Ft_1Acc, D_Ct_3Acc, D_Ct_2Acc, D_Ct_1Acc, D_Ot_3Acc, D_Ot_2Acc, D_Ot_1Acc, 
            D_ct_3Acc, D_ct_2Acc, D_ct_1Acc, D_ot_3Acc, D_ot_2Acc, D_ot_1Acc, D_ft_3Acc, D_ft_2Acc, D_ft_1Acc, D_it_3Acc, D_it_2Acc, D_it_1Acc], \
        _ = theano.scan(fn=self.backwardPass,          
                                    sequences = [TytAcc, TYtAcc, T_egOutMB,
                                                 self.p1(TIt_3Acc), self.p1(TIt_2Acc), self.p1(TIt_1Acc), self.p1(TFt_3Acc), self.p1(TFt_2Acc), self.p1(TFt_1Acc), 
                                                    self.p1(TOt_3Acc), self.p1(TOt_2Acc), self.p1(TOt_1Acc), self.p1(TCt_3Acc), self.p1(TCt_2Acc), self.p1(TCt_1Acc),
                                                 TIt_3Acc, TIt_2Acc, TIt_1Acc, TFt_3Acc, TFt_2Acc, TFt_1Acc, TOt_3Acc, TOt_2Acc, TOt_1Acc, TCt_3Acc, TCt_2Acc, TCt_1Acc,
                                                 self.d1(Tc3Acc), self.d1(Tc2Acc), self.d1(Tc1Acc),
                                                 self.p1(Tft_3Acc), self.p1(Tft_2Acc), self.p1(Tft_1Acc), self.p1(Tit_3Acc), self.p1(Tit_2Acc), self.p1(Tit_1Acc),
                                                 Tit_3Acc, Tit_2Acc, Tit_1Acc, Tot_3Acc, Tot_2Acc, Tot_1Acc, Tc3Acc, Tc2Acc, Tc1Acc],
                                    outputs_info=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, 
                                                    cp.deepcopy(h3v), cp.deepcopy(h2v), cp.deepcopy(h1v), 
                                                    cp.deepcopy(h3v), cp.deepcopy(h2v), cp.deepcopy(h1v), 
                                                    cp.deepcopy(h3v), cp.deepcopy(h2v), cp.deepcopy(h1v),
                                                    cp.deepcopy(h3v), cp.deepcopy(h2v), cp.deepcopy(h1v)],
                                                    go_backwards=True)



        ###### calculate the sum of weight errors (gradient) over time on all weights: ######
        [Wxi1Acc, Wxi2Acc, Wxi3Acc, Wxf1Acc, Wxf2Acc, Wxf3Acc, Wxo1Acc, Wxo2Acc, Wxo3Acc, Wxc1Acc, Wxc2Acc, Wxc3Acc,
         Whi1Acc, Whi2Acc, Whi3Acc, Whf1Acc, Whf2Acc, Whf3Acc, Who1Acc, Who2Acc, Who3Acc, Whc1Acc, Whc2Acc, Whc3Acc,
         Why1Acc, Why2Acc, Why3Acc,
         Wxj2Acc, Wxj3Acc, Wfj2Acc, Wfj3Acc, Wcj2Acc, Wcj3Acc, Woj2Acc, Woj3Acc,
         Wci1Acc, Wci2Acc, Wci3Acc, Wcf1Acc, Wcf2Acc, Wcf3Acc, Wco1Acc, Wco2Acc, Wco3Acc], \
        _ = theano.scan(fn=self.gradSum, 
                                               sequences = [T_egMB, self.d1(Th1Acc), self.d1(Th2Acc), self.d1(Th3Acc), 
                                                            D_It_1Acc, D_It_2Acc, D_It_3Acc, D_Ft_1Acc, D_Ft_2Acc, D_Ft_3Acc, D_Ot_1Acc, D_Ot_2Acc, D_Ot_3Acc, D_Ct_1Acc, D_Ct_2Acc, D_Ct_3Acc,
                                                            D_YtAcc, Th1Acc, Th2Acc, Th3Acc, self.d1(Tc1Acc), self.d1(Tc2Acc), self.d1(Tc3Acc), Tc1Acc, Tc2Acc, Tc3Acc],
                                               outputs_info = [cp.deepcopy(ioh1), cp.deepcopy(ioh2), cp.deepcopy(ioh3), cp.deepcopy(ioh1), cp.deepcopy(ioh2), cp.deepcopy(ioh3), 	
                                                                cp.deepcopy(ioh1), cp.deepcopy(ioh2), cp.deepcopy(ioh3), cp.deepcopy(ioh1), cp.deepcopy(ioh2), cp.deepcopy(ioh3),
                                                                cp.deepcopy(h1h1), cp.deepcopy(h2h2), cp.deepcopy(h3h3), cp.deepcopy(h1h1), cp.deepcopy(h2h2), cp.deepcopy(h3h3), 
                                                                cp.deepcopy(h1h1), cp.deepcopy(h2h2), cp.deepcopy(h3h3), cp.deepcopy(h1h1), cp.deepcopy(h2h2), cp.deepcopy(h3h3),
                                                                cp.deepcopy(h1io), cp.deepcopy(h2io), cp.deepcopy(h3io),
                                                                cp.deepcopy(h1h2), cp.deepcopy(h2h3), cp.deepcopy(h1h2), cp.deepcopy(h2h3), cp.deepcopy(h1h2), cp.deepcopy(h2h3), cp.deepcopy(h1h2), cp.deepcopy(h2h3),
                                                                cp.deepcopy(h1v), cp.deepcopy(h2v), cp.deepcopy(h3v), cp.deepcopy(h1v), cp.deepcopy(h2v), cp.deepcopy(h3v), cp.deepcopy(h1v), cp.deepcopy(h2v), cp.deepcopy(h3v)])



        return [Wxi1Acc[-1], Wxi2Acc[-1], Wxi3Acc[-1], Wxf1Acc[-1], Wxf2Acc[-1], Wxf3Acc[-1], Wxo1Acc[-1], Wxo2Acc[-1], Wxo3Acc[-1], Wxc1Acc[-1], Wxc2Acc[-1], Wxc3Acc[-1],
                 Whi1Acc[-1], Whi2Acc[-1], Whi3Acc[-1], Whf1Acc[-1], Whf2Acc[-1], Whf3Acc[-1], Who1Acc[-1], Who2Acc[-1], Who3Acc[-1], Whc1Acc[-1], Whc2Acc[-1], Whc3Acc[-1],
                 Why1Acc[-1], Why2Acc[-1], Why3Acc[-1],
                 Wxj2Acc[-1], Wxj3Acc[-1], Wfj2Acc[-1], Wfj3Acc[-1], Wcj2Acc[-1], Wcj3Acc[-1], Woj2Acc[-1], Woj3Acc[-1],
                 Wci1Acc[-1], Wci2Acc[-1], Wci3Acc[-1], Wcf1Acc[-1], Wcf2Acc[-1], Wcf3Acc[-1], Wco1Acc[-1], Wco2Acc[-1], Wco3Acc[-1],
                T.sum(D_It_1Acc, axis=0, acc_dtype=theano.config.floatX), T.sum(D_It_2Acc, axis=0, acc_dtype=theano.config.floatX),
                T.sum(D_It_3Acc, axis=0, acc_dtype=theano.config.floatX), T.sum(D_Ft_1Acc, axis=0, acc_dtype=theano.config.floatX),
                T.sum(D_Ft_2Acc, axis=0, acc_dtype=theano.config.floatX), T.sum(D_Ft_3Acc, axis=0, acc_dtype=theano.config.floatX),
                T.sum(D_Ct_1Acc, axis=0, acc_dtype=theano.config.floatX), T.sum(D_Ct_2Acc, axis=0, acc_dtype=theano.config.floatX),
                T.sum(D_Ct_3Acc, axis=0, acc_dtype=theano.config.floatX), T.sum(D_Ot_1Acc, axis=0, acc_dtype=theano.config.floatX),
                T.sum(D_Ot_2Acc, axis=0, acc_dtype=theano.config.floatX), T.sum(D_Ot_3Acc, axis=0, acc_dtype=theano.config.floatX),
                T.sum(D_YtAcc, axis=0, acc_dtype=theano.config.floatX), 
                T.sum(KLAcc, acc_dtype=theano.config.floatX), Th1Acc[-1], Th2Acc[-1], Th3Acc[-1], Tc1Acc[-1], Tc2Acc[-1], Tc3Acc[-1]]
        

    def mean(self, inputT):
        '''
        auxiliary function to calculatemean gradient, which is a matrix
        input is an array of matrix
        '''
        return T.mean(inputT, axis=0, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)


       
    def train(self, dataset, noOfEpochPerMB, noOfEpoch, sizeOfMiniBatch, lengthOfMB):
        '''
        given input and output examples, trains the 3 layer LSTM using grad descent and RMSprop


        egMatrix, egoutMatrix: rows of examples, rows are supposed to be grouped into minibatches before feed into compiled gradient fn
                                e.g. egMatrix(rows1-10), egMatrix(100-110) etc... ramdomize the minibatches
                                each call to the compiled gradient fn will give you a single update on all of the weights

        arguments:

        egMatrix (np.array): 2d array of input example notes over time, Index as:  [#row(time index) in example] [#note in exmaple]  
                            for minibatch training, each minibatch have to the the same size (no. of rows-time steps) and it will be randomised
                            total number of input notes (cols) at each time step (rows) have to be the same as the self.io_length parameter

        egoutMatrix (np.array): 2d array of multiple output example notes over time, index as  [#row(time index) in example] [#note in exmaple]
                    normally egoutMatrix is simply a time delayed version of egMatrix
                    for minibatch training, each minibatch have to occupy the same number of time steps (rows) in the example Matrixes
                    total number of output notes (cols) at each time step (rows) have to be the same as the self.io_length parameter

        noOfEpoch (int): how many times you want to run through the input examples for training

        sizeOfMiniBatch (int): the input example can be broken into further miniBatches. Each miniBatch induces a single gradient update per epoch. 
                                the mini. batches will be randomised during training.

        '''

        T_egMB = T.tensor3(name='T_egMB', dtype=theano.config.floatX)
        T_egOutMB = T.tensor3(name='T_egOutMB', dtype=theano.config.floatX)	
                              

        [Wxi1AccR, Wxi2AccR, Wxi3AccR, Wxf1AccR, Wxf2AccR, Wxf3AccR, Wxo1AccR, Wxo2AccR, Wxo3AccR, Wxc1AccR, Wxc2AccR, Wxc3AccR,
         Whi1AccR, Whi2AccR, Whi3AccR, Whf1AccR, Whf2AccR, Whf3AccR, Who1AccR, Who2AccR, Who3AccR, Whc1AccR, Whc2AccR, Whc3AccR,
         Why1AccR, Why2AccR, Why3AccR,
         Wxj2AccR, Wxj3AccR, Wfj2AccR, Wfj3AccR, Wcj2AccR, Wcj3AccR, Woj2AccR, Woj3AccR,
         Wci1AccR, Wci2AccR, Wci3AccR, Wcf1AccR, Wcf2AccR, Wcf3AccR, Wco1AccR, Wco2AccR, Wco3AccR,
         D_It_1AccR, D_It_2AccR, D_It_3AccR, D_Ft_1AccR, D_Ft_2AccR, D_Ft_3AccR, D_Ct_1AccR, D_Ct_2AccR, D_Ct_3AccR, D_Ot_1AccR, D_Ot_2AccR, D_Ot_3AccR,
         D_YtAccR, KLAccR,  Th1AccR, Th2AccR, Th3AccR, Tc1AccR, Tc2AccR, Tc3AccR], scan_update4 = theano.scan(fn=self.gradSingleTune, sequences=[T_egMB, T_egOutMB],
                                                                                                    outputs_info = [None, None, None, None, None, None, None, None, None, 
                                                                                                                    None, None, None, None, None, None, None, None, None, 
                                                                                                                    None, None, None, None, None, None, None, None, None, 
                                                                                                                     None, None, None, None, None, None, None, None, 
                                                                                                                     None, None, None, None, None, None, None, None, None, 
                                                                                                                     None, None, None, None, None, None, None, None, None, None, None, None, 
                                                                                                                     None, None, self.htd1_1, self.htd1_2, self.htd1_3, self.ctd1_1, self.ctd1_2, self.ctd1_3])
                                                                                                                     
        #blah mod output_info above, or inside gradSingleTune such that each MB example you start from zero states


        
        #RMS prop update gradient of all weights with previously saved gradient and newly summed (over time) gradient. newly summed gradients have magnitudes clipped to self.gradClip  
        DWxi1 = self.RMSgrad(self.DWxi1p, self.gClip(self.mean(Wxi1AccR)));  DWxi2 = self.RMSgrad(self.DWxi2p, self.gClip(self.mean(Wxi2AccR)));  DWxi3 = self.RMSgrad(self.DWxi3p, self.gClip(self.mean(Wxi3AccR)))
        DWxf1 = self.RMSgrad(self.DWxf1p, self.gClip(self.mean(Wxf1AccR)));  DWxf2 = self.RMSgrad(self.DWxf2p, self.gClip(self.mean(Wxf2AccR)));  DWxf3 = self.RMSgrad(self.DWxf3p, self.gClip(self.mean(Wxf3AccR)))
        DWxo1 = self.RMSgrad(self.DWxo1p, self.gClip(self.mean(Wxo1AccR)));  DWxo2 = self.RMSgrad(self.DWxo2p, self.gClip(self.mean(Wxo2AccR)));  DWxo3 = self.RMSgrad(self.DWxo3p, self.gClip(self.mean(Wxo3AccR)))
        DWxc1 = self.RMSgrad(self.DWxc1p, self.gClip(self.mean(Wxc1AccR)));  DWxc2 = self.RMSgrad(self.DWxc2p, self.gClip(self.mean(Wxc2AccR)));  DWxc3 = self.RMSgrad(self.DWxc3p, self.gClip(self.mean(Wxc3AccR)))
        
        DWhi1 = self.RMSgrad(self.DWhi1p, self.gClip(self.mean(Whi1AccR)));  DWhi2 = self.RMSgrad(self.DWhi2p, self.gClip(self.mean(Whi2AccR)));  DWhi3 = self.RMSgrad(self.DWhi3p, self.gClip(self.mean(Whi3AccR)))
        DWhf1 = self.RMSgrad(self.DWhf1p, self.gClip(self.mean(Whf1AccR)));  DWhf2 = self.RMSgrad(self.DWhf2p, self.gClip(self.mean(Whf2AccR)));  DWhf3 = self.RMSgrad(self.DWhf3p, self.gClip(self.mean(Whf3AccR)))
        DWho1 = self.RMSgrad(self.DWho1p, self.gClip(self.mean(Who1AccR)));  DWho2 = self.RMSgrad(self.DWho2p, self.gClip(self.mean(Who2AccR)));  DWho3 = self.RMSgrad(self.DWho3p, self.gClip(self.mean(Who3AccR)))
        DWhc1 = self.RMSgrad(self.DWhc1p, self.gClip(self.mean(Whc1AccR)));  DWhc2 = self.RMSgrad(self.DWhc2p, self.gClip(self.mean(Whc2AccR)));  DWhc3 = self.RMSgrad(self.DWhc3p, self.gClip(self.mean(Whc3AccR)))
        
        DWhy1 = self.RMSgrad(self.DWhy1p, self.gClip(self.mean(Why1AccR)));  DWhy2 = self.RMSgrad(self.DWhy2p, self.gClip(self.mean(Why2AccR)));  DWhy3 = self.RMSgrad(self.DWhy3p, self.gClip(self.mean(Why3AccR)))

        DWxj2 = self.RMSgrad(self.DWxj2p, self.gClip(self.mean(Wxj2AccR)));  DWxj3 = self.RMSgrad(self.DWxj3p, self.gClip(self.mean(Wxj3AccR)))
        DWfj2 = self.RMSgrad(self.DWfj2p, self.gClip(self.mean(Wfj2AccR)));  DWfj3 = self.RMSgrad(self.DWfj3p, self.gClip(self.mean(Wfj3AccR)))
        DWcj2 = self.RMSgrad(self.DWcj2p, self.gClip(self.mean(Wcj2AccR)));  DWcj3 = self.RMSgrad(self.DWcj3p, self.gClip(self.mean(Wcj3AccR)))
        DWoj2 = self.RMSgrad(self.DWoj2p, self.gClip(self.mean(Woj2AccR)));  DWoj3 = self.RMSgrad(self.DWoj3p, self.gClip(self.mean(Woj3AccR)))
        
        DWci1 = self.RMSgrad(self.DWci1p, self.gClip(self.mean(Wci1AccR)));  DWci2 = self.RMSgrad(self.DWci2p, self.gClip(self.mean(Wci2AccR)));  DWci3 = self.RMSgrad(self.DWci3p, self.gClip(self.mean(Wci3AccR)))
        DWcf1 = self.RMSgrad(self.DWcf1p, self.gClip(self.mean(Wcf1AccR)));  DWcf2 = self.RMSgrad(self.DWcf2p, self.gClip(self.mean(Wcf2AccR)));  DWcf3 = self.RMSgrad(self.DWcf3p, self.gClip(self.mean(Wcf3AccR)))
        DWco1 = self.RMSgrad(self.DWco1p, self.gClip(self.mean(Wco1AccR)));  DWco2 = self.RMSgrad(self.DWco2p, self.gClip(self.mean(Wco2AccR)));  DWco3 = self.RMSgrad(self.DWco3p, self.gClip(self.mean(Wco3AccR)))

        Dbi1 = self.RMSgrad(self.Dbi1p, self.gClip(self.mean(D_It_1AccR)))
        Dbi2 = self.RMSgrad(self.Dbi2p, self.gClip(self.mean(D_It_2AccR)))
        Dbi3 = self.RMSgrad(self.Dbi3p, self.gClip(self.mean(D_It_3AccR)))
        Dbf1 = self.RMSgrad(self.Dbf1p, self.gClip(self.mean(D_Ft_1AccR)))
        Dbf2 = self.RMSgrad(self.Dbf2p, self.gClip(self.mean(D_Ft_2AccR)))
        Dbf3 = self.RMSgrad(self.Dbf3p, self.gClip(self.mean(D_Ft_3AccR)))
        Dbc1 = self.RMSgrad(self.Dbc1p, self.gClip(self.mean(D_Ct_1AccR)))
        Dbc2 = self.RMSgrad(self.Dbc2p, self.gClip(self.mean(D_Ct_2AccR)))
        Dbc3 = self.RMSgrad(self.Dbc3p, self.gClip(self.mean(D_Ct_3AccR)))
        Dbo1 = self.RMSgrad(self.Dbo1p, self.gClip(self.mean(D_Ot_1AccR)))
        Dbo2 = self.RMSgrad(self.Dbo2p, self.gClip(self.mean(D_Ot_2AccR)))
        Dbo3 = self.RMSgrad(self.Dbo3p, self.gClip(self.mean(D_Ot_3AccR)))
        Dby = self.RMSgrad(self.Dbyp, self.gClip(self.mean(D_YtAccR)))
        
        

        
        
        #compile theano function for gradient updates based on bk prop and RMSprop gradient calculations:       
        gradfn = theano.function(inputs=[T_egMB, T_egOutMB],
                                 outputs = [DWxi1, DWxi2, DWxi3, DWxf1, DWxf2, DWxf3, DWxo1, DWxo2, DWxo3, DWxc1, DWxc2, DWxc3,
                                            DWhi1, DWhi2, DWhi3, DWhf1, DWhf2, DWhf3, DWho1, DWho2, DWho3, DWhc1, DWhc2, DWhc3,
                                            DWhy1, DWhy2, DWhy3,
                                            DWxj2, DWxj3, DWfj2, DWfj3, DWcj2, DWcj3, DWoj2, DWoj3,
                                            DWci1, DWci2, DWci3, DWcf1, DWcf2, DWcf3, DWco1, DWco2, DWco3, KLAccR,
                                            Dbi1, Dbi2, Dbi3, Dbf1, Dbf2, Dbf3, Dbc1, Dbc2, Dbc3, Dbo1, Dbo2, Dbo3, Dby,
                                            Th1AccR, Th2AccR, Th3AccR, Tc1AccR, Tc2AccR, Tc3AccR],
                                 allow_input_downcast = True, 
                                #updates with instance variables "self.ud1 + self.ud2 + self.ud3" also works too... 
                                 updates = scan_update4 + [(self.Wxi_1, self.Wxi_1 - self.R1*DWxi1), (self.Wxi_2, self.Wxi_2 - self.R2*DWxi2), (self.Wxi_3, self.Wxi_3 - self.R3*DWxi3),
                                        (self.Wxf_1, self.Wxf_1 - self.R1*DWxf1), (self.Wxf_2, self.Wxf_2 - self.R2*DWxf2), (self.Wxf_3, self.Wxf_3 - self.R3*DWxf3),
                                        (self.Wxo_1, self.Wxo_1 - self.R1*DWxo1), (self.Wxo_2, self.Wxo_2 - self.R2*DWxo2), (self.Wxo_3, self.Wxo_3 - self.R3*DWxo3),
                                        (self.Wxc_1, self.Wxc_1 - self.R1*DWxc1), (self.Wxc_2, self.Wxc_2 - self.R2*DWxc2), (self.Wxc_3, self.Wxc_3 - self.R3*DWxc3),
                                        (self.Whi_1, self.Whi_1 - self.R1*DWhi1), (self.Whi_2, self.Whi_2 - self.R2*DWhi2), (self.Whi_3, self.Whi_3 - self.R3*DWhi3),
                                        (self.Whf_1, self.Whf_1 - self.R1*DWhf1), (self.Whf_2, self.Whf_2 - self.R2*DWhf2), (self.Whf_3, self.Whf_3 - self.R3*DWhf3),
                                        (self.Who_1, self.Who_1 - self.R1*DWho1), (self.Who_2, self.Who_2 - self.R2*DWho2), (self.Who_3, self.Who_3 - self.R3*DWho3),
                                        (self.Whc_1, self.Whc_1 - self.R1*DWhc1), (self.Whc_2, self.Whc_2 - self.R2*DWhc2), (self.Whc_3, self.Whc_3 - self.R3*DWhc3),
                                        (self.Why_1, self.Why_1 - self.R1*DWhy1), (self.Why_2, self.Why_2 - self.R2*DWhy2), (self.Why_3, self.Why_3 - self.R3*DWhy3),
                                        (self.Wxj_2, self.Wxj_2 - self.R2*DWxj2), (self.Wxj_3, self.Wxj_3 - self.R3*DWxj3),
                                        (self.Wfj_2, self.Wfj_2 - self.R2*DWfj2), (self.Wfj_3, self.Wfj_3 - self.R3*DWfj3),
                                        (self.Wcj_2, self.Wcj_2 - self.R2*DWcj2), (self.Wcj_3, self.Wcj_3 - self.R3*DWcj3),
                                        (self.Woj_2, self.Woj_2 - self.R2*DWoj2), (self.Woj_3, self.Woj_3 - self.R3*DWoj3),
                                        (self.Wci_1, self.Wci_1 - self.R1*DWci1), (self.Wci_2, self.Wci_2 - self.R2*DWci2), (self.Wci_3, self.Wci_3 - self.R3*DWci3),
                                        (self.Wcf_1, self.Wcf_1 - self.R1*DWcf1), (self.Wcf_2, self.Wcf_2 - self.R2*DWcf2), (self.Wcf_3, self.Wcf_3 - self.R3*DWcf3),
                                        (self.Wco_1, self.Wco_1 - self.R1*DWco1), (self.Wco_2, self.Wco_2 - self.R2*DWco2), (self.Wco_3, self.Wco_3 - self.R3*DWco3),
                                        (self.loss, self.mean(KLAccR)),
                                        (self.bi_1, self.bi_1 - self.R1*Dbi1), (self.bi_2, self.bi_2 - self.R2*Dbi2), (self.bi_3, self.bi_3 - self.R3*Dbi3),
                                        (self.bf_1, self.bf_1 - self.R1*Dbf1), (self.bf_2, self.bf_2 - self.R2*Dbf2), (self.bf_3, self.bf_3 - self.R3*Dbf3),
                                        (self.bc_1, self.bc_1 - self.R1*Dbc1), (self.bc_2, self.bc_2 - self.R2*Dbc2), (self.bc_3, self.bc_3 - self.R3*Dbc3),
                                        (self.bo_1, self.bo_1 - self.R1*Dbo1), (self.bo_2, self.bo_2 - self.R2*Dbo2), (self.bo_3, self.bo_3 - self.R3*Dbo3),
                                        (self.by, self.by - self.Rout*Dby),
                                        (self.DWxi1p, DWxi1), (self.DWxi2p, DWxi2), (self.DWxi3p, DWxi3),
                                        (self.DWxf1p, DWxf1), (self.DWxf2p, DWxf2), (self.DWxf3p, DWxf3),
                                        (self.DWxo1p, DWxo1), (self.DWxo2p, DWxo2), (self.DWxo3p, DWxo3),
                                        (self.DWxc1p, DWxc1), (self.DWxc2p, DWxc2), (self.DWxc3p, DWxc3),
                                        (self.DWhi1p, DWhi1), (self.DWhi2p, DWhi2), (self.DWhi3p, DWhi3),
                                        (self.DWhf1p, DWhf1), (self.DWhf2p, DWhf2), (self.DWhf3p, DWhf3),
                                        (self.DWho1p, DWho1), (self.DWho2p, DWho2), (self.DWho3p, DWho3),
                                        (self.DWhc1p, DWhc1), (self.DWhc2p, DWhc2), (self.DWhc3p, DWhc3),
                                        (self.DWhy1p, DWhy1), (self.DWhy2p, DWhy2), (self.DWhy3p, DWhy3),
                                        (self.DWxj2p, DWxj2), (self.DWxj3p, DWxj3),
                                        (self.DWfj2p, DWfj2), (self.DWfj3p, DWfj3),
                                        (self.DWcj2p, DWcj2), (self.DWcj3p, DWcj3),
                                        (self.DWoj2p, DWoj2), (self.DWoj3p, DWoj3),
                                        (self.DWci1p, DWci1), (self.DWci2p, DWci2), (self.DWci3p, DWci3),
                                        (self.DWcf1p, DWcf1), (self.DWcf2p, DWcf2), (self.DWcf3p, DWcf3),
                                        (self.DWco1p, DWco1), (self.DWco2p, DWco2), (self.DWco3p, DWco3),
                                        (self.Dbi1p, Dbi1), (self.Dbi2p, Dbi2), (self.Dbi3p, Dbi3),
                                        (self.Dbf1p, Dbf1), (self.Dbf2p, Dbf2), (self.Dbf3p, Dbf3),
                                        (self.Dbc1p, Dbc1), (self.Dbc2p, Dbc2), (self.Dbc3p, Dbc3),
                                        (self.Dbo1p, Dbo1), (self.Dbo2p, Dbo2), (self.Dbo3p, Dbo3),
                                        (self.Dbyp, Dby), 
                                        (self.ctd1_1, Tc1AccR[-1]), (self.ctd1_2, Tc2AccR[-1]), (self.ctd1_3, Tc3AccR[-1]),
                                        (self.htd1_1, Th1AccR[-1]), (self.htd1_2, Th2AccR[-1]), (self.htd1_3, Th3AccR[-1])
                                        ], mode = 'FAST_RUN')
        
        #useful prints function structure to file:
        #theano.printing.pydotprint(gradfn)    

        dataLength = [np.array(dataset[n]).shape[0] for n in np.arange(np.array(dataset).shape[0])]
        tuneLength = lengthOfMB #np.max(dataLength)
        tuneIndex = np.arange(np.array(dataset).shape[0])

        print("-------------------------------------------")
        print("no. of epochs/MB = " + str(noOfEpochPerMB))
        print("size of mini-batch = " + str(sizeOfMiniBatch))
        print("no. of epochs of MB = " + str(noOfEpoch))
        print("length of each tune in MB = " +str(tuneLength))
        print("-------------------------------------------")


        for n in np.arange(noOfEpoch):
                shuffle(tuneIndex)
                mbDataset = []
                print("Epoch: " + str(n))
                print("tunes in mini-batch: " + str(tuneIndex[0:sizeOfMiniBatch]))
                for n in np.arange(0,sizeOfMiniBatch,1): #np.array(dataset).shape[0],1):
                        #make sure all tunes are same length as longest tune by replicating itselves so the MB mean gradient is fair!
                        if (dataLength[tuneIndex[n]] < lengthOfMB) :
                            mbDataset.append(np.concatenate((((dataset[tuneIndex[n]].tolist())*int(tuneLength/(dataLength[tuneIndex[n]]))), dataset[tuneIndex[n]][0:(tuneLength%dataLength[tuneIndex[n]])])))
                        else:
                            mbDataset.append(dataset[tuneIndex[n]][0:lengthOfMB])
                        #plt.figure(n)
                        #plt.imshow(np.transpose(mbDataset[n]), origin='lower', aspect='auto',
                        #     interpolation='nearest', cmap=pylab.cm.gray_r)
                        #plt.colorbar()
                plt.show()
                for m in np.arange(noOfEpochPerMB):
                        pretime = time.time()
                        #self.printDerivatives()
                        #self.printWeights()
                        #print("checking size of input tensor = " + str(np.array(mbDataset)[0:sizeOfMiniBatch,0:tuneLength-2,:].shape) + ", and " + str(np.array(mbDataset)[0:sizeOfMiniBatch,1:tuneLength-1,:].shape))
                        returns = gradfn(np.float32(np.array(mbDataset)[0:sizeOfMiniBatch,0:tuneLength-2,:]), np.float32(np.array(mbDataset)[0:sizeOfMiniBatch,1:tuneLength-1,:])) 
                        print("         time taken = " + str(time.time()-pretime) + ", loss = " + str(np.asarray(self.loss.eval())))



        #prevLoss = 0
        #MBindex = np.arange(0, len(egMatrix), sizeOfMiniBatch)#len(egMatrix)#start, end, step
        #print("minibatch split: " + str(MBindex))
        #pretime=time.time()
        #for j in xrange(noOfEpoch):
        #    #np.random.shuffle(MBindex)
        #    #self.resetStates()
        #    #self.resetRMSgrads()
        #    #print('epoch ' + str(j))
        #    for k in MBindex:
        #        #self.resetStates()                
        #        returns = gradfn(egMatrix[k:k+sizeOfMiniBatch], egoutMatrix[k:k+sizeOfMiniBatch])
        #        print("time taken = " + str(time.time()-pretime) + ", loss = " + str(np.asarray(self.loss.eval())) + ", minibatch " + str(k) + ", epoch " + str(j))
                
    
    def printDerivatives(self):
        '''
        prints all weight derivatives (as saved between updates for RMSprop)
        '''
        
        print('DWxi1p=' + str(np.array(self.DWxi1p.eval()))); print('DWxi2p=' + str(np.array(self.DWxi2p.eval()))); print('DWxi3p=' + str(np.array(self.DWxi3p.eval())));
        print('DWxf1p=' + str(np.array(self.DWxf1p.eval()))); print('DWxf2p=' + str(np.array(self.DWxf2p.eval()))); print('DWxf3p=' + str(np.array(self.DWxf3p.eval())));
        print('DWxo1p=' + str(np.array(self.DWxo1p.eval()))); print('DWxo2p=' + str(np.array(self.DWxo2p.eval()))); print('DWxo3p=' + str(np.array(self.DWxo3p.eval())));
        print('DWxc1p=' + str(np.array(self.DWxc1p.eval()))); print('DWxc2p=' + str(np.array(self.DWxc2p.eval()))); print('DWxc3p=' + str(np.array(self.DWxc3p.eval())));
        print('DWhi1p=' + str(np.array(self.DWhi1p.eval()))); print('DWhi2p=' + str(np.array(self.DWhi2p.eval()))); print('DWhi3p=' + str(np.array(self.DWhi3p.eval())));
        print('DWhf1p=' + str(np.array(self.DWhf1p.eval()))); print('DWhf2p=' + str(np.array(self.DWhf2p.eval()))); print('DWhf3p=' + str(np.array(self.DWhf3p.eval())));
        print('DWho1p=' + str(np.array(self.DWho1p.eval()))); print('DWho2p=' + str(np.array(self.DWho2p.eval()))); print('DWho3p=' + str(np.array(self.DWho3p.eval())));
        print('DWhc1p=' + str(np.array(self.DWhc1p.eval()))); print('DWhc2p=' + str(np.array(self.DWhc2p.eval()))); print('DWhc3p=' + str(np.array(self.DWhc3p.eval())));
        print('DWhy1p=' + str(np.array(self.DWhy1p.eval()))); print('DWhy2p=' + str(np.array(self.DWhy2p.eval()))); print('DWhy3p=' + str(np.array(self.DWhy3p.eval())));
        
        print('DWxj2p=' + str(np.array(self.DWxj2p.eval()))); print('DWxj3p=' + str(np.array(self.DWxj3p.eval())));
        print('DWfj2p=' + str(np.array(self.DWfj2p.eval()))); print('DWfj3p=' + str(np.array(self.DWfj3p.eval())));
        print('DWcj2p=' + str(np.array(self.DWcj2p.eval()))); print('DWcj3p=' + str(np.array(self.DWcj3p.eval())));
        print('DWoj2p=' + str(np.array(self.DWoj2p.eval()))); print('DWoj3p=' + str(np.array(self.DWoj3p.eval())));
        
        print('DWci1p=' + str(np.array(self.DWci1p.eval()))); print('DWci2p=' + str(np.array(self.DWci2p.eval()))); print('DWxi3p=' + str(np.array(self.DWci3p.eval())));
        print('DWcf1p=' + str(np.array(self.DWcf1p.eval()))); print('DWcf2p=' + str(np.array(self.DWcf2p.eval()))); print('DWxf3p=' + str(np.array(self.DWcf3p.eval())));
        print('DWco1p=' + str(np.array(self.DWco1p.eval()))); print('DWco2p=' + str(np.array(self.DWco2p.eval()))); print('DWxo3p=' + str(np.array(self.DWco3p.eval())));
        
        print('Dbi1p=' + str(np.array(self.Dbi1p.eval()))); print('Dbi2p=' + str(np.array(self.Dbi2p.eval()))); print('Dbi3p=' + str(np.array(self.Dbi3p.eval())));
        print('Dbf1p=' + str(np.array(self.Dbf1p.eval()))); print('Dbf2p=' + str(np.array(self.Dbf2p.eval()))); print('Dbf3p=' + str(np.array(self.Dbf3p.eval())));
        print('Dbc1p=' + str(np.array(self.Dbc1p.eval()))); print('Dbc2p=' + str(np.array(self.Dbc2p.eval()))); print('Dbc3p=' + str(np.array(self.Dbc3p.eval())));
        print('Dbo1p=' + str(np.array(self.Dbo1p.eval()))); print('Dbo2p=' + str(np.array(self.Dbo2p.eval()))); print('Dbo3p=' + str(np.array(self.Dbo3p.eval())));
        print('Dbyp=' + str(np.array(self.Dbyp.eval())));
        
        
    def printWeights(self):
        '''
        prints all weight matrixes
        '''
        
        print('Wxi1=' + str(np.array(self.Wxi_1.eval()))); print('Wxi2=' + str(np.array(self.Wxi_2.eval()))); print('Wxi3=' + str(np.array(self.Wxi_3.eval())));
        print('Wxf1=' + str(np.array(self.Wxf_1.eval()))); print('Wxf2=' + str(np.array(self.Wxf_2.eval()))); print('Wxf3=' + str(np.array(self.Wxf_3.eval())));
        print('Wxo1=' + str(np.array(self.Wxo_1.eval()))); print('Wxo2=' + str(np.array(self.Wxo_2.eval()))); print('Wxo3=' + str(np.array(self.Wxo_3.eval())));
        print('Wxc1=' + str(np.array(self.Wxc_1.eval()))); print('Wxc2=' + str(np.array(self.Wxc_2.eval()))); print('Wxc3=' + str(np.array(self.Wxc_3.eval())));
        print('Whi1=' + str(np.array(self.Whi_1.eval()))); print('Whi2=' + str(np.array(self.Whi_2.eval()))); print('Whi3=' + str(np.array(self.Whi_3.eval())));
        print('Whf1=' + str(np.array(self.Whf_1.eval()))); print('Whf2=' + str(np.array(self.Whf_2.eval()))); print('Whf3=' + str(np.array(self.Whf_3.eval())));
        print('Who1=' + str(np.array(self.Who_1.eval()))); print('Who2=' + str(np.array(self.Who_2.eval()))); print('Who3=' + str(np.array(self.Who_3.eval())));
        print('Whc1=' + str(np.array(self.Whc_1.eval()))); print('Whc2=' + str(np.array(self.Whc_2.eval()))); print('Whc3=' + str(np.array(self.Whc_3.eval())));
        print('Why1=' + str(np.array(self.Why_1.eval()))); print('Why2=' + str(np.array(self.Why_2.eval()))); print('Why3=' + str(np.array(self.Why_3.eval())));
        
        print('Wxj2=' + str(np.array(self.Wxj_2.eval()))); print('Wxj3=' + str(np.array(self.Wxj_3.eval())));
        print('Wfj2=' + str(np.array(self.Wfj_2.eval()))); print('Wfj3=' + str(np.array(self.Wfj_3.eval())));
        print('Wcj2=' + str(np.array(self.Wcj_2.eval()))); print('Wcj3=' + str(np.array(self.Wcj_3.eval())));
        print('Woj2=' + str(np.array(self.Woj_2.eval()))); print('Woj3=' + str(np.array(self.Woj_3.eval())));
        
        print('Wci1=' + str(np.array(self.Wci_1.eval()))); print('Wci2=' + str(np.array(self.Wci_2.eval()))); print('Wxi3=' + str(np.array(self.Wci_3.eval())));
        print('Wcf1=' + str(np.array(self.Wcf_1.eval()))); print('Wcf2=' + str(np.array(self.Wcf_2.eval()))); print('Wxf3=' + str(np.array(self.Wcf_3.eval())));
        print('Wco1=' + str(np.array(self.Wco_1.eval()))); print('Wco2=' + str(np.array(self.Wco_2.eval()))); print('Wxo3=' + str(np.array(self.Wco_3.eval())));
        
        print('bi1=' + str(np.array(self.bi_1.eval()))); print('bi2=' + str(np.array(self.bi_2.eval()))); print('bi3=' + str(np.array(self.bi_3.eval())));
        print('bf1=' + str(np.array(self.bf_1.eval()))); print('bf2=' + str(np.array(self.bf_2.eval()))); print('bf3=' + str(np.array(self.bf_3.eval())));
        print('bc1=' + str(np.array(self.bc_1.eval()))); print('bc2=' + str(np.array(self.bc_2.eval()))); print('bc3=' + str(np.array(self.bc_3.eval())));
        print('bo1=' + str(np.array(self.bo_1.eval()))); print('bo2=' + str(np.array(self.bo_2.eval()))); print('bo3=' + str(np.array(self.bo_3.eval())));
        print('by=' + str(np.array(self.by.eval())));


    def setNewRates(self, R1, R2, R3, Rout):
        self.R1 = np.float32(R1)
        self.R2 = np.float32(R2)
        self.R3 = np.float32(R3)
        self.Rout = np.float32(Rout)


    def genMusic(self, startVectors, noOfSamps):
        '''
        gemerates music by feeding output of network back into input of network at next time step
        example is ran on the network before music is generated with the feedback

        arguments:
            startVectors (np.array): each row of startVectors represent a time step. the midi notes are assigned across the columns i.e. startVectors[#time][#midi note]

            noOfSamps (int): number of samples (time steps) to generate after running through the startVector example
        '''
                
        T_startVectors = T.matrix(name='T_startVectors', dtype=theano.config.floatX)
        h1fdbk = T.vector(name='h1fdbk', dtype=theano.config.floatX)
        h2fdbk = T.vector(name='h2fdbk', dtype=theano.config.floatX)
        h3fdbk = T.vector(name='h3fdbk', dtype=theano.config.floatX)
        c1fdbk = T.vector(name='c1fdbk', dtype=theano.config.floatX)
        c2fdbk = T.vector(name='c2fdbk', dtype=theano.config.floatX)
        c3fdbk = T.vector(name='c3fdbk', dtype=theano.config.floatX)
        
        [nextStartAcc1, nextStartAcc2, preh1Acc, preh2Acc, preh3Acc, prec1Acc, prec2Acc, prec3Acc,
         preit1Acc, preit2Acc, preit3Acc, preft1Acc, preft2Acc, preft3Acc, 
         preot1Acc, preot2Acc, preot3Acc, preIt1Acc, preIt2Acc, preIt3Acc,
         preFt1Acc, preFt2Acc, preFt3Acc, preOt1Acc, preOt2Acc, preOt3Acc, 
         preCt1Acc, preCt2Acc, preCt3Acc], pre_scan_updates = theano.scan(fn=self.forwardPass, sequences = [T_startVectors],
                                               outputs_info = [None, None, h1fdbk, h2fdbk, h3fdbk, c1fdbk, c2fdbk, c3fdbk, None, None, None, None, None, None, None, None, None,
                                                                None, None, None, None, None, None, None, None, None, None, None, None])
        
        startProb = nextStartAcc1[-1] #/ T.sum(nextStartAcc1[-1],acc_dtype=theano.config.floatX)  #No mormalisation, use outputvalue as probability will be good enough!?
        startSamples = self.T_rng.binomial(size=(1,self.io_length), p=startProb, dtype=theano.config.floatX)
        
        [genResults1, genResults2, genh1Acc, genh2Acc, genh3Acc, genc1Acc, genc2Acc, genc3Acc,
         genit1Acc, genit2Acc, genit3Acc, genft1Acc, genft2Acc, genft3Acc, 
         genot1Acc, genot2Acc, genot3Acc, genIt1Acc, genIt2Acc, genIt3Acc,
         genFt1Acc, genFt2Acc, genFt3Acc, genOt1Acc, genOt2Acc, genOt3Acc, 
         genCt1Acc, genCt2Acc, genCt3Acc, genOutProb], scan_updates_gen = theano.scan(fn=self.forwardPassGen, 
                                               outputs_info = [startSamples[0], None, preh1Acc[-1], preh2Acc[-1], preh3Acc[-3], prec1Acc[-1], prec2Acc[-1], prec3Acc[-1], 
                                                               None, None, None, None, None, None, None, None, None,
                                                                None, None, None, None, None, None, None, None, None, None, None, None, None], n_steps=noOfSamps)

        genfn = theano.function(inputs=[T_startVectors], outputs=[genResults1, genOutProb], 
                                allow_input_downcast=True,
                                givens = {h1fdbk:self.htd1_1, h2fdbk:self.htd1_2, h3fdbk:self.htd1_3, c1fdbk:self.ctd1_1, c2fdbk:self.ctd1_2, c3fdbk:self.ctd1_3},
                                          updates= pre_scan_updates + scan_updates_gen)

        return genfn(startVectors)
    

        

