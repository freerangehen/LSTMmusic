############## music generation with 3 layer LSTM ######################
#
#           Distributed under the MIT license by Henry Ip
########################################################################


import pydot
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
from random import shuffle
import theano 
from theano import tensor as T, function, printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.sandbox.cuda as cuda
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
        self.muRand = 0.01 # for weights/biases that are initialisaed to ~uniform(0, muRand) 
        self.sigRand = 0.1 #S.D. of normally distributed initialisation of weights/biases

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
            
        outProb = gyt #/ T.sum(gyt, acc_dtype=theano.config.floatX) 
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

        D_yt_temp = - (kt / T.maximum(self.epsilon, yt)) + ((np.float32(1.0) - kt) / (1-T.minimum((np.float32(1.0) - self.epsilon),yt))) #cross entropy cost function
        D_yt = D_yt_temp #cross entropy square error
        #D_yt_temp = yt - kt #sqr error cost function
        #D_yt = D_yt_temp[0] #sqr error cost function
        D_Yt = T.mul(D_yt, self.gdot(Yt))
        
        #sqrCost = T.sum(T.mul(np.float32(0.5), T.mul(kt-yt,kt-yt)), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
        sqrCost = T.sum(-T.mul(kt,T.log(yt)) - T.mul((np.float32(1.0)-kt), T.log(np.float32(1.0)-yt)))


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
               
        return [sqrCost, D_Yt, D_It_3, D_It_2, D_It_1, D_Ft_3, D_Ft_2, D_Ft_1, D_Ct_3, D_Ct_2, D_Ct_1, D_Ot_3, D_Ot_2, D_Ot_1, 
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

    def gClip_np(self, inMatrix):
        
        return np.clip(inMatrix, -self.gradClip, self.gradClip)
    
        
    def tanh(self, inVec):
        '''
        computes tanh(input), input is theano tensor (symbolic)
        '''
        return T.mul(np.float32(2.0), T.nnet.sigmoid(T.mul(np.float32(2.0), inVec))) - np.float32(1.0)
        
    def RMSgrad(self, prevGrad, newGrad):
        gradSqr = T.mul(np.float32(0.9), T.mul(prevGrad, prevGrad)) + T.mul(np.float32(0.1), T.mul(newGrad, newGrad))
        return (newGrad / T.sqrt(T.maximum(self.epsilon, gradSqr)))

    def RMSgrad_np(self, prevGrad, newGrad):
        gradSqr = np.float32(0.9)*(prevGrad**2) + np.float32(0.1)*(newGrad**2)
        return newGrad / np.sqrt(np.clip(gradSqr, self.epsilon, gradSqr))
    
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
        (theano tensor.matrix)	: Accumulated gradient of weights.

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
        
       
    def train(self, dataset, noOfEpoch, MBn, lengthOfMB):
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

        MBn (int): the input example can be broken into further miniBatches. Each miniBatch induces a single gradient update per epoch. 
                                the mini. batches will be randomised during training.

        '''

        # some local variables: 
        T_0_5 = theano.shared(value=np.float32(0.5),name = 'T_0_5', allow_downcast=True)
        T_egMB = T.matrix(name='T_egMB', dtype=theano.config.floatX)
        T_egOutMB = T.matrix(name='T_egOutMB', dtype=theano.config.floatX)
        Tin = T.vector(name='T_in', dtype=theano.config.floatX)
        #output_info temporary symbols for forward run:
        Thtd1_1 = T.vector(name='Thtd1_1', dtype=theano.config.floatX);  Thtd1_2 = T.vector(name='Thtd1_2', dtype=theano.config.floatX);  Thtd1_3 = T.vector(name='Thtd1_3', dtype=theano.config.floatX)
        Tctd1_1 = T.vector(name='Tctd1_1', dtype=theano.config.floatX);  Tctd1_2 = T.vector(name='Tctd1_2', dtype=theano.config.floatX);  Tctd1_3 = T.vector(name='Tctd1_3', dtype=theano.config.floatX)
        #output_info temporary symbols for backwards run:
        TDct3 = T.vector(name='TDct3', dtype=theano.config.floatX);  TDct2 = T.vector(name='TDct2', dtype=theano.config.floatX);  TDct1 = T.vector(name='TDct1', dtype=theano.config.floatX)
        TDot3 = T.vector(name='TDot3', dtype=theano.config.floatX);  TDot2 = T.vector(name='TDot2', dtype=theano.config.floatX);  TDot1 = T.vector(name='TDot1', dtype=theano.config.floatX)
        TDft3 = T.vector(name='TDft3', dtype=theano.config.floatX);  TDft2 = T.vector(name='TDft2', dtype=theano.config.floatX);  TDft1 = T.vector(name='TDft1', dtype=theano.config.floatX)
        TDit3 = T.vector(name='TDit3', dtype=theano.config.floatX);  TDit2 = T.vector(name='TDit2', dtype=theano.config.floatX);  TDit1 = T.vector(name='TDit1', dtype=theano.config.floatX)
        #weight gradient accumlation run temporary symbols
        TWxi1 = T.matrix(name='TWxi1', dtype=theano.config.floatX);  TWxi2 = T.matrix(name='TWxi2', dtype=theano.config.floatX);  TWxi3 = T.matrix(name='TWxi3', dtype=theano.config.floatX)
        TWxf1 = T.matrix(name='TWxf1', dtype=theano.config.floatX);  TWxf2 = T.matrix(name='TWxf2', dtype=theano.config.floatX);  TWxf3 = T.matrix(name='TWxf3', dtype=theano.config.floatX)
        TWxo1 = T.matrix(name='TWxo1', dtype=theano.config.floatX);  TWxo2 = T.matrix(name='TWxo2', dtype=theano.config.floatX);  TWxo3 = T.matrix(name='TWxo3', dtype=theano.config.floatX) 
        TWxc1 = T.matrix(name='TWxc1', dtype=theano.config.floatX);  TWxc2 = T.matrix(name='TWxc2', dtype=theano.config.floatX);  TWxc3 = T.matrix(name='TWxc3', dtype=theano.config.floatX)  
        
        TWhi1 = T.matrix(name='TWhi1', dtype=theano.config.floatX);  TWhi2 = T.matrix(name='TWhi2', dtype=theano.config.floatX);  TWhi3 = T.matrix(name='TWhi3', dtype=theano.config.floatX)
        TWhf1 = T.matrix(name='TWhf1', dtype=theano.config.floatX);  TWhf2 = T.matrix(name='TWhf2', dtype=theano.config.floatX);  TWhf3 = T.matrix(name='TWhf3', dtype=theano.config.floatX)
        TWho1 = T.matrix(name='TWho1', dtype=theano.config.floatX);  TWho2 = T.matrix(name='TWho2', dtype=theano.config.floatX);  TWho3 = T.matrix(name='TWho3', dtype=theano.config.floatX) 
        TWhc1 = T.matrix(name='TWhc1', dtype=theano.config.floatX);  TWhc2 = T.matrix(name='TWhc2', dtype=theano.config.floatX);  TWhc3 = T.matrix(name='TWhc3', dtype=theano.config.floatX) 
        
        TWhy1 = T.matrix(name='TWhy1', dtype=theano.config.floatX);  TWhy2 = T.matrix(name='TWhy2', dtype=theano.config.floatX);  TWhy3 = T.matrix(name='TWhy3', dtype=theano.config.floatX) 
        
        TWxj2 = T.matrix(name='TWxj2', dtype=theano.config.floatX);  TWxj3 = T.matrix(name='TWxj3', dtype=theano.config.floatX)
        TWfj2 = T.matrix(name='TWfj2', dtype=theano.config.floatX);  TWfj3 = T.matrix(name='TWfj3', dtype=theano.config.floatX)
        TWcj2 = T.matrix(name='TWcj2', dtype=theano.config.floatX);  TWcj3 = T.matrix(name='TWcj3', dtype=theano.config.floatX)
        TWoj2 = T.matrix(name='TWoj2', dtype=theano.config.floatX);  TWoj3 = T.matrix(name='TWoj3', dtype=theano.config.floatX)
        
        TWci1 = T.vector(name='TWci1', dtype=theano.config.floatX);  TWci2 = T.vector(name='TWci2', dtype=theano.config.floatX);  TWci3 = T.vector(name='TWci3', dtype=theano.config.floatX)
        TWcf1 = T.vector(name='TWcf1', dtype=theano.config.floatX);  TWcf2 = T.vector(name='TWcf2', dtype=theano.config.floatX);  TWcf3 = T.vector(name='TWcf3', dtype=theano.config.floatX)
        TWco1 = T.vector(name='TWco1', dtype=theano.config.floatX);  TWco2 = T.vector(name='TWco2', dtype=theano.config.floatX);  TWco3 = T.vector(name='TWco3', dtype=theano.config.floatX)


        ###### forward run ######         
        #usage of scan(): sequences will fill up input arguments first, the leftover arguments will be filled by "output_infos". outputs_info should match return pattern, 
        # with ones not feeding back marked as 'None'
        [TytAcc, TYtAcc, Th1Acc, Th2Acc, Th3Acc, Tc1Acc, Tc2Acc, Tc3Acc, 
         Tit_1Acc, Tit_2Acc, Tit_3Acc, Tft_1Acc, Tft_2Acc, Tft_3Acc, Tot_1Acc, Tot_2Acc, Tot_3Acc,
         TIt_1Acc, TIt_2Acc, TIt_3Acc, TFt_1Acc, TFt_2Acc, TFt_3Acc, TOt_1Acc, TOt_2Acc, TOt_3Acc, TCt_1Acc, TCt_2Acc, TCt_3Acc], \
        scan_updates = theano.scan(fn=self.forwardPass, 
                                    outputs_info=[None, None, Thtd1_1, Thtd1_2, Thtd1_3, Tctd1_1, Tctd1_2, Tctd1_3, 
                                                  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], 
                                    sequences=[T_egMB]) 
        
        ###### forward run recheck if rows are indeed stuck... ######
        
        #forwardfn = theano.function(inputs=[T_egMB],
        #                         outputs = [TytAcc, TYtAcc, Th1Acc, Th2Acc, Th3Acc, Tc1Acc, Tc2Acc, Tc3Acc, 
        #                                     Tit_1Acc, Tit_2Acc, Tit_3Acc, Tft_1Acc, Tft_2Acc, Tft_3Acc, Tot_1Acc, Tot_2Acc, Tot_3Acc,
        #                                     TIt_1Acc, TIt_2Acc, TIt_3Acc, TFt_1Acc, TFt_2Acc, TFt_3Acc, TOt_1Acc, TOt_2Acc, TOt_3Acc, TCt_1Acc, TCt_2Acc, TCt_3Acc],
        #                         givens = {Thtd1_1:self.htd1_1, Thtd1_2:self.htd1_2, Thtd1_3:self.htd1_3, Tctd1_1:self.ctd1_1, Tctd1_2:self.ctd1_2, Tctd1_3:self.ctd1_3}, 
        #                           allow_input_downcast = True, updates = scan_updates, mode = 'FAST_RUN')
        #
        #forwardResults = forwardfn(egMatrix[0:11])
        #
        #print('forward TytAcc = ' + str(np.array(forwardResults[0]))); print('forward TYtAcc = ' + str(np.array(forwardResults[1]))); print('forward Th1Acc = ' + str(np.array(forwardResults[2])))
        #print('forward Th2Acc = ' + str(np.array(forwardResults[3]))); print('forward Th3Acc = ' + str(np.array(forwardResults[4]))); print('forward Tc1Acc = ' + str(np.array(forwardResults[5])))
        #print('forward Tc2Acc = ' + str(np.array(forwardResults[6]))); print('forward Tc3Acc = ' + str(np.array(forwardResults[7]))); print('forward Tit_1Acc = ' + str(np.array(forwardResults[8])))
        #print('forward Tit_2Acc = ' + str(np.array(forwardResults[9]))); print('forward Tit_3Acc = ' + str(np.array(forwardResults[10]))); print('forward Tft_1Acc = ' + str(np.array(forwardResults[11])))
        #print('forward Tft_2Acc = ' + str(np.array(forwardResults[12]))); print('forward Tft_3Acc = ' + str(np.array(forwardResults[13]))); print('forward Tot_1Acc = ' + str(np.array(forwardResults[14])))
        #print('forward Tot_2Acc = ' + str(np.array(forwardResults[15]))); print('forward Tot_3Acc = ' + str(np.array(forwardResults[16])))
                
        

        ###### backwards run: generating error signals for all variables across all layers and across time ######
        [KLAcc, D_YtAcc, D_It_3Acc, D_It_2Acc, D_It_1Acc, D_Ft_3Acc, D_Ft_2Acc, D_Ft_1Acc, D_Ct_3Acc, D_Ct_2Acc, D_Ct_1Acc, D_Ot_3Acc, D_Ot_2Acc, D_Ot_1Acc, 
            D_ct_3Acc, D_ct_2Acc, D_ct_1Acc, D_ot_3Acc, D_ot_2Acc, D_ot_1Acc, D_ft_3Acc, D_ft_2Acc, D_ft_1Acc, D_it_3Acc, D_it_2Acc, D_it_1Acc], \
        scan_updates2 = theano.scan(fn=self.backwardPass,          
                                    sequences = [TytAcc, TYtAcc, T_egOutMB,
                                                 self.p1(TIt_3Acc), self.p1(TIt_2Acc), self.p1(TIt_1Acc), self.p1(TFt_3Acc), self.p1(TFt_2Acc), self.p1(TFt_1Acc), self.p1(TOt_3Acc), self.p1(TOt_2Acc), self.p1(TOt_1Acc), self.p1(TCt_3Acc), self.p1(TCt_2Acc), self.p1(TCt_1Acc),
                                                 TIt_3Acc, TIt_2Acc, TIt_1Acc, TFt_3Acc, TFt_2Acc, TFt_1Acc, TOt_3Acc, TOt_2Acc, TOt_1Acc, TCt_3Acc, TCt_2Acc, TCt_1Acc,
                                                 self.d1(Tc3Acc), self.d1(Tc2Acc), self.d1(Tc1Acc),
                                                 self.p1(Tft_3Acc), self.p1(Tft_2Acc), self.p1(Tft_1Acc), self.p1(Tit_3Acc), self.p1(Tit_2Acc), self.p1(Tit_1Acc),
                                                 Tit_3Acc, Tit_2Acc, Tit_1Acc, Tot_3Acc, Tot_2Acc, Tot_1Acc, Tc3Acc, Tc2Acc, Tc1Acc],
                                    outputs_info=[None, None, None, None, None, None, None, None, None, None, None, None, None, None, 
                                                    TDct3, TDct2, TDct1, TDot3, TDot2, TDot1, TDft3, TDft2, TDft1, TDit3, TDit2, TDit1],
                                    go_backwards=True)
        
        ###### calculate the sum of weight errors (gradient) over time on all weights: ######
        [Wxi1Acc, Wxi2Acc, Wxi3Acc, Wxf1Acc, Wxf2Acc, Wxf3Acc, Wxo1Acc, Wxo2Acc, Wxo3Acc, Wxc1Acc, Wxc2Acc, Wxc3Acc,
         Whi1Acc, Whi2Acc, Whi3Acc, Whf1Acc, Whf2Acc, Whf3Acc, Who1Acc, Who2Acc, Who3Acc, Whc1Acc, Whc2Acc, Whc3Acc,
         Why1Acc, Why2Acc, Why3Acc,
         Wxj2Acc, Wxj3Acc, Wfj2Acc, Wfj3Acc, Wcj2Acc, Wcj3Acc, Woj2Acc, Woj3Acc,
         Wci1Acc, Wci2Acc, Wci3Acc, Wcf1Acc, Wcf2Acc, Wcf3Acc, Wco1Acc, Wco2Acc, Wco3Acc], \
        scan_updates3 = theano.scan(fn=self.gradSum, 
                                               sequences = [T_egMB, self.d1(Th1Acc), self.d1(Th2Acc), self.d1(Th3Acc), 
                                                            D_It_1Acc, D_It_2Acc, D_It_3Acc, D_Ft_1Acc, D_Ft_2Acc, D_Ft_3Acc, D_Ot_1Acc, D_Ot_2Acc, D_Ot_3Acc, D_Ct_1Acc, D_Ct_2Acc, D_Ct_3Acc,
                                                            D_YtAcc, Th1Acc, Th2Acc, Th3Acc, self.d1(Tc1Acc), self.d1(Tc2Acc), self.d1(Tc3Acc), Tc1Acc, Tc2Acc, Tc3Acc],
                                               outputs_info = [TWxi1, TWxi2, TWxi3, TWxf1, TWxf2, TWxf3, TWxo1, TWxo2, TWxo3, TWxc1, TWxc2, TWxc3,
                                                               TWhi1, TWhi2, TWhi3, TWhf1, TWhf2, TWhf3, TWho1, TWho2, TWho3, TWhc1, TWhc2, TWhc3,
                                                               TWhy1, TWhy2, TWhy3,
                                                               TWxj2, TWxj3, TWfj2, TWfj3, TWcj2, TWcj3, TWoj2, TWoj3,
                                                               TWci1, TWci2, TWci3, TWcf1, TWcf2, TWcf3, TWco1, TWco2, TWco3])
        

        


        singleGradfn = theano.function(inputs=[T_egMB, T_egOutMB,
                                         TDct3, TDct2, TDct1, TDot3, TDot2, TDot1, TDft3, TDft2, TDft1, TDit3, TDit2, TDit1],
                                 outputs = [Wxi1Acc[-1], Wxi2Acc[-1], Wxi3Acc[-1], Wxf1Acc[-1], Wxf2Acc[-1], Wxf3Acc[-1],
                                            Wxo1Acc[-1], Wxo2Acc[-1], Wxo3Acc[-1], Wxc1Acc[-1], Wxc2Acc[-1], Wxc3Acc[-1],        
                                            Whi1Acc[-1], Whi2Acc[-1], Whi3Acc[-1], Whf1Acc[-1], Whf2Acc[-1], Whf3Acc[-1],
                                            Who1Acc[-1], Who2Acc[-1], Who3Acc[-1], Whc1Acc[-1], Whc2Acc[-1], Whc3Acc[-1],
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
                                            Th1Acc[-1], Th2Acc[-1], Th3Acc[-1], Tc1Acc[-1], Tc2Acc[-1], Tc3Acc[-1],
                                            TytAcc, Th1Acc, D_YtAcc, D_It_3Acc, D_It_2Acc, D_It_1Acc, D_Ft_3Acc, D_Ft_2Acc, D_Ft_1Acc, D_Ct_3Acc, D_Ct_2Acc, D_Ct_1Acc, D_Ot_3Acc, D_Ot_2Acc, D_Ot_1Acc, 
                                            D_ct_3Acc, D_ct_2Acc, D_ct_1Acc, D_ot_3Acc, D_ot_2Acc, D_ot_1Acc, D_ft_3Acc, D_ft_2Acc, D_ft_1Acc, D_it_3Acc, D_it_2Acc, D_it_1Acc,
                                            T.sum(KLAcc, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)],
                                 givens = {Thtd1_1:self.htd1_1, Thtd1_2:self.htd1_2, Thtd1_3:self.htd1_3, Tctd1_1:self.ctd1_1, Tctd1_2:self.ctd1_2, Tctd1_3:self.ctd1_3,
                                           TWxi1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxi2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxi3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
                                           TWxf1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxf2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxf3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
                                           TWxo1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxo2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxo3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
                                           TWxc1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxc2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxc3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
                                           TWhi1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWhi2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWhi3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
                                           TWhf1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWhf2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWhf3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
                                           TWho1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWho2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWho3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
                                           TWhc1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWhc2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWhc3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
                                           TWhy1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.io_length)])), TWhy2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.io_length)])), TWhy3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.io_length)])),
                                           TWxj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWxj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
                                           TWfj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWfj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
                                           TWcj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWcj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
                                           TWoj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWoj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
                                           TWci1:np.float32(np.asarray([0.0]*self.h1_length)), TWci2:np.float32(np.asarray([0.0]*self.h2_length)), TWci3:np.float32(np.asarray([0.0]*self.h3_length)),
                                           TWcf1:np.float32(np.asarray([0.0]*self.h1_length)), TWcf2:np.float32(np.asarray([0.0]*self.h2_length)), TWcf3:np.float32(np.asarray([0.0]*self.h3_length)),
                                           TWco1:np.float32(np.asarray([0.0]*self.h1_length)), TWco2:np.float32(np.asarray([0.0]*self.h2_length)), TWco3:np.float32(np.asarray([0.0]*self.h3_length))
                                           },
                                 allow_input_downcast = True, 
                                 updates = scan_updates + scan_updates2 + scan_updates3 + \
                                                        [(self.ctd1_1, Tc1Acc[-1]), (self.ctd1_2, Tc2Acc[-1]), (self.ctd1_3, Tc3Acc[-1]),
                                                         (self.htd1_1, Th1Acc[-1]), (self.htd1_2, Th2Acc[-1]), (self.htd1_3, Th3Acc[-1])], 
                                 mode = 'FAST_RUN')
     
        
        #compile theano function for gradient updates based on bk prop and RMSprop gradient calculations:       
        #gradfn = theano.function(inputs=[T_egMB, T_egOutMB,
        #                                 TDct3, TDct2, TDct1, TDot3, TDot2, TDot1, TDft3, TDft2, TDft1, TDit3, TDit2, TDit1],
        #                         outputs = [TytAcc, Th1Acc, D_YtAcc, D_It_3Acc, D_It_2Acc, D_It_1Acc, D_Ft_3Acc, D_Ft_2Acc, D_Ft_1Acc, D_Ct_3Acc, D_Ct_2Acc, D_Ct_1Acc, D_Ot_3Acc, D_Ot_2Acc, D_Ot_1Acc, 
        #                                    D_ct_3Acc, D_ct_2Acc, D_ct_1Acc, D_ot_3Acc, D_ot_2Acc, D_ot_1Acc, D_ft_3Acc, D_ft_2Acc, D_ft_1Acc, D_it_3Acc, D_it_2Acc, D_it_1Acc, 
        #                                    DWxi1, DWxi2, DWxi3, DWxf1, DWxf2, DWxf3, DWxo1, DWxo2, DWxo3, DWxc1, DWxc2, DWxc3,
        #                                    DWhi1, DWhi2, DWhi3, DWhf1, DWhf2, DWhf3, DWho1, DWho2, DWho3, DWhc1, DWhc2, DWhc3,
        #                                    DWhy1, DWhy2, DWhy3,
        #                                    DWxj2, DWxj3, DWfj2, DWfj3, DWcj2, DWcj3, DWoj2, DWoj3,
        #                                    DWci1, DWci2, DWci3, DWcf1, DWcf2, DWcf3, DWco1, DWco2, DWco3, KLAcc,
        #                                    Dbi1, Dbi2, Dbi3, Dbf1, Dbf2, Dbf3, Dbc1, Dbc2, Dbc3, Dbo1, Dbo2, Dbo3, Dby,
        #                                    Th1Acc[-1], Th2Acc[-1], Th3Acc[-1], Tc1Acc[-1], Tc2Acc[-1], Tc3Acc[-1]],
        #                         givens = {Thtd1_1:self.htd1_1, Thtd1_2:self.htd1_2, Thtd1_3:self.htd1_3, Tctd1_1:self.ctd1_1, Tctd1_2:self.ctd1_2, Tctd1_3:self.ctd1_3,
        #                                   TWxi1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxi2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxi3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
        #                                   TWxf1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxf2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxf3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
        #                                   TWxo1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxo2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxo3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
        #                                   TWxc1:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h1_length)])), TWxc2:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h2_length)])), TWxc3:np.float32(np.asarray([[0.0]*self.io_length for n in range(self.h3_length)])),
        #                                   TWhi1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWhi2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWhi3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
        #                                   TWhf1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWhf2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWhf3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
        #                                   TWho1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWho2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWho3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
        #                                   TWhc1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h1_length)])), TWhc2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h2_length)])), TWhc3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.h3_length)])),
        #                                   TWhy1:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.io_length)])), TWhy2:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.io_length)])), TWhy3:np.float32(np.asarray([[0.0]*self.h3_length for n in range(self.io_length)])),
        #                                   TWxj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWxj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
        #                                   TWfj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWfj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
        #                                   TWcj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWcj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
        #                                   TWoj2:np.float32(np.asarray([[0.0]*self.h1_length for n in range(self.h2_length)])), TWoj3:np.float32(np.asarray([[0.0]*self.h2_length for n in range(self.h3_length)])),
        #                                   TWci1:np.float32(np.asarray([0.0]*self.h1_length)), TWci2:np.float32(np.asarray([0.0]*self.h2_length)), TWci3:np.float32(np.asarray([0.0]*self.h3_length)),
        #                                   TWcf1:np.float32(np.asarray([0.0]*self.h1_length)), TWcf2:np.float32(np.asarray([0.0]*self.h2_length)), TWcf3:np.float32(np.asarray([0.0]*self.h3_length)),
        #                                   TWco1:np.float32(np.asarray([0.0]*self.h1_length)), TWco2:np.float32(np.asarray([0.0]*self.h2_length)), TWco3:np.float32(np.asarray([0.0]*self.h3_length))
        #                                   },
        #                         allow_input_downcast = True, updates = scan_updates + scan_updates2 + scan_updates3 + \
        #                         [(self.Wxi_1, self.Wxi_1 - self.R1*DWxi1), (self.Wxi_2, self.Wxi_2 - self.R2*DWxi2), (self.Wxi_3, self.Wxi_3 - self.R3*DWxi3),
        #                          (self.Wxf_1, self.Wxf_1 - self.R1*DWxf1), (self.Wxf_2, self.Wxf_2 - self.R2*DWxf2), (self.Wxf_3, self.Wxf_3 - self.R3*DWxf3),
        #                          (self.Wxo_1, self.Wxo_1 - self.R1*DWxo1), (self.Wxo_2, self.Wxo_2 - self.R2*DWxo2), (self.Wxo_3, self.Wxo_3 - self.R3*DWxo3),
        #                          (self.Wxc_1, self.Wxc_1 - self.R1*DWxc1), (self.Wxc_2, self.Wxc_2 - self.R2*DWxc2), (self.Wxc_3, self.Wxc_3 - self.R3*DWxc3),
        #                          (self.Whi_1, self.Whi_1 - self.R1*DWhi1), (self.Whi_2, self.Whi_2 - self.R2*DWhi2), (self.Whi_3, self.Whi_3 - self.R3*DWhi3),
        #                          (self.Whf_1, self.Whf_1 - self.R1*DWhf1), (self.Whf_2, self.Whf_2 - self.R2*DWhf2), (self.Whf_3, self.Whf_3 - self.R3*DWhf3),
        #                          (self.Who_1, self.Who_1 - self.R1*DWho1), (self.Who_2, self.Who_2 - self.R2*DWho2), (self.Who_3, self.Who_3 - self.R3*DWho3),
        #                          (self.Whc_1, self.Whc_1 - self.R1*DWhc1), (self.Whc_2, self.Whc_2 - self.R2*DWhc2), (self.Whc_3, self.Whc_3 - self.R3*DWhc3),
        #                          (self.Why_1, self.Why_1 - self.R1*DWhy1), (self.Why_2, self.Why_2 - self.R2*DWhy2), (self.Why_3, self.Why_3 - self.R3*DWhy3),
        #                          (self.Wxj_2, self.Wxj_2 - self.R2*DWxj2), (self.Wxj_3, self.Wxj_3 - self.R3*DWxj3),
        #                          (self.Wfj_2, self.Wfj_2 - self.R2*DWfj2), (self.Wfj_3, self.Wfj_3 - self.R3*DWfj3),
        #                          (self.Wcj_2, self.Wcj_2 - self.R2*DWcj2), (self.Wcj_3, self.Wcj_3 - self.R3*DWcj3),
        #                          (self.Woj_2, self.Woj_2 - self.R2*DWoj2), (self.Woj_3, self.Woj_3 - self.R3*DWoj3),
        #                          (self.Wci_1, self.Wci_1 - self.R1*DWci1), (self.Wci_2, self.Wci_2 - self.R2*DWci2), (self.Wci_3, self.Wci_3 - self.R3*DWci3),
        #                          (self.Wcf_1, self.Wcf_1 - self.R1*DWcf1), (self.Wcf_2, self.Wcf_2 - self.R2*DWcf2), (self.Wcf_3, self.Wcf_3 - self.R3*DWcf3),
        #                          (self.Wco_1, self.Wco_1 - self.R1*DWco1), (self.Wco_2, self.Wco_2 - self.R2*DWco2), (self.Wco_3, self.Wco_3 - self.R3*DWco3),
        #                          (self.loss, T.sum(KLAcc, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)),
        #                          (self.bi_1, self.bi_1 - self.R1*Dbi1), (self.bi_2, self.bi_2 - self.R2*Dbi2), (self.bi_3, self.bi_3 - self.R3*Dbi3),
        #                          (self.bf_1, self.bf_1 - self.R1*Dbf1), (self.bf_2, self.bf_2 - self.R2*Dbf2), (self.bf_3, self.bf_3 - self.R3*Dbf3),
        #                          (self.bc_1, self.bc_1 - self.R1*Dbc1), (self.bc_2, self.bc_2 - self.R2*Dbc2), (self.bc_3, self.bc_3 - self.R3*Dbc3),
        #                          (self.bo_1, self.bo_1 - self.R1*Dbo1), (self.bo_2, self.bo_2 - self.R2*Dbo2), (self.bo_3, self.bo_3 - self.R3*Dbo3),
        #                          (self.by, self.by - self.Rout*Dby),
        #                          (self.DWxi1p, Wxi1Acc[-1]), (self.DWxi2p, DWxi2), (self.DWxi3p, DWxi3),
        #                          (self.DWxf1p, DWxf1), (self.DWxf2p, DWxf2), (self.DWxf3p, DWxf3),
        #                          (self.DWxo1p, DWxo1), (self.DWxo2p, DWxo2), (self.DWxo3p, DWxo3),
        #                          (self.DWxc1p, DWxc1), (self.DWxc2p, DWxc2), (self.DWxc3p, DWxc3),
        #                          (self.DWhi1p, Whi1Acc[-1]), (self.DWhi2p, DWhi2), (self.DWhi3p, DWhi3),
        #                          (self.DWhf1p, DWhf1), (self.DWhf2p, DWhf2), (self.DWhf3p, DWhf3),
        #                          (self.DWho1p, DWho1), (self.DWho2p, DWho2), (self.DWho3p, DWho3),
        #                          (self.DWhc1p, DWhc1), (self.DWhc2p, DWhc2), (self.DWhc3p, DWhc3),
        #                          (self.DWhy1p, DWhy1), (self.DWhy2p, DWhy2), (self.DWhy3p, DWhy3),
        #                          (self.DWxj2p, DWxj2), (self.DWxj3p, DWxj3),
        #                          (self.DWfj2p, DWfj2), (self.DWfj3p, DWfj3),
        #                          (self.DWcj2p, DWcj2), (self.DWcj3p, DWcj3),
        #                          (self.DWoj2p, DWoj2), (self.DWoj3p, DWoj3),
        #                          (self.DWci1p, DWci1), (self.DWci2p, DWci2), (self.DWci3p, DWci3),
        #                          (self.DWcf1p, DWcf1), (self.DWcf2p, DWcf2), (self.DWcf3p, DWcf3),
        #                          (self.DWco1p, DWco1), (self.DWco2p, DWco2), (self.DWco3p, DWco3),
        #                          (self.Dbi1p, Dbi1), (self.Dbi2p, Dbi2), (self.Dbi3p, Dbi3),
        #                          (self.Dbf1p, Dbf1), (self.Dbf2p, Dbf2), (self.Dbf3p, Dbf3),
        #                          (self.Dbc1p, Dbc1), (self.Dbc2p, Dbc2), (self.Dbc3p, Dbc3),
        #                          (self.Dbo1p, Dbo1), (self.Dbo2p, Dbo2), (self.Dbo3p, Dbo3),
        #                          (self.Dbyp, Dby), 
        #                          (self.ctd1_1, Tc1Acc[-1]), (self.ctd1_2, Tc2Acc[-1]), (self.ctd1_3, Tc3Acc[-1]),
        #                          (self.htd1_1, Th1Acc[-1]), (self.htd1_2, Th2Acc[-1]), (self.htd1_3, Th3Acc[-1])
        #                          ], 
        #                         mode = 'FAST_RUN')
        
        #useful prints function structure to file:
        #theano.printing.pydotprint(gradfn)    



        dataLength = [np.array(dataset[n]).shape[0] for n in np.arange(np.array(dataset).shape[0])]
        tuneLength = lengthOfMB #np.max(dataLength)
        tuneIndex = np.arange(np.array(dataset).shape[0])


        print("-------------------------------------------")
        print("size of mini-batch = " + str(MBn))
        print("no. of epochs of MB = " + str(noOfEpoch))
        print("length of each tune in MB = " +str(tuneLength))
        print("-------------------------------------------")



        for m in np.arange(noOfEpoch):
            pretime=time.time()
            epochLoss = np.float32(0.0)
            shuffle(tuneIndex)
            mbDataset = []
            print("Epoch: " + str(n))
            print("tunes in mini-batch: " + str(tuneIndex[0:MBn]))
            for n in np.arange(0,MBn,1): #np.array(dataset).shape[0],1):
                #make sure all tunes are same length as longest tune by replicating itselves so the MB mean gradient is fair!
                if (dataLength[tuneIndex[n]] < lengthOfMB) :
                    mbDataset.append(np.concatenate((((dataset[tuneIndex[n]].tolist())*int(tuneLength/(dataLength[tuneIndex[n]]))), dataset[tuneIndex[n]][0:(tuneLength%dataLength[tuneIndex[n]])])))
                else:
                    mbDataset.append(dataset[tuneIndex[n]][0:lengthOfMB])
                #    plt.figure(n)
                #    plt.imshow(np.transpose(mbDataset[n]), origin='lower', aspect='auto',
                #                 interpolation='nearest', cmap=pylab.cm.gray_r)
                #    plt.colorbar()
                #plt.show()

            ###### variables for gradient mean ###### 
            #input gate:
            dWxi1=np.array(np.float32([[0.0]*self.io_length for n in range(self.h1_length)])); dWhi1=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
            dWci1=np.array(np.float32([0.0]*self.h1_length)); dbi1=np.array(np.float32([0.0]*self.h1_length))
            #forget gate:
            dWxf1=np.array(np.float32([[0.0]*self.io_length for n in range(self.h1_length)])); dWhf1=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
            dWcf1=np.array(np.float32([0.0]*self.h1_length)); dbf1=np.array(np.float32([0.0]*self.h1_length))
            #output gate:
            dWxo1=np.array(np.float32([[0.0]*self.io_length for n in range(self.h1_length)])); dWho1=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
            dWco1=np.array(np.float32([0.0]*self.h1_length)); dbo1=np.array(np.float32([0.0]*self.h1_length))
            #state:
            dWxc1=np.array(np.float32([[0.0]*self.io_length for n in range(self.h1_length)])); dWhc1=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h1_length)]))
            dbc1=np.array(np.float32([0.0]*self.h1_length))
            #input gate:
            dWxi2=np.array(np.float32([[0.0]*self.io_length for n in range(self.h2_length)])); dWxj2=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
            dWhi2=np.array(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)])); dWci2=np.array(np.float32([0.0]*self.h2_length)); dbi2=np.array(np.float32([0.0]*self.h2_length))
            #forget gate:
            dWxf2=np.array(np.float32([[0.0]*self.io_length for n in range(self.h2_length)])); dWfj2=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
            dWhf2=np.array(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)])); dWcf2=np.array(np.float32([0.0]*self.h2_length)); dbf2=np.array(np.float32([0.0]*self.h2_length))
            #output gate:
            dWxo2=np.array(np.float32([[0.0]*self.io_length for n in range(self.h2_length)])); dWoj2=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
            dWho2=np.array(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)])); dWco2=np.array(np.float32([0.0]*self.h2_length)); dbo2=np.array(np.float32([0.0]*self.h2_length))
            #state:
            dWxc2=np.array(np.float32([[0.0]*self.io_length for n in range(self.h2_length)])); dWcj2=np.array(np.float32([[0.0]*self.h1_length for n in range(self.h2_length)]))
            dWhc2=np.array(np.float32([[0.0]*self.h2_length for n in range(self.h2_length)])); dbc2=np.array(np.float32([0.0]*self.h2_length))
            ###### third layer RMSprop previous gradients ###### 
            #input gate:
            dWxi3=np.array(np.float32([[0.0]*self.io_length for n in range(self.h3_length)])); dWxj3=np.array(np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]))
            dWhi3=np.array(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)])); dWci3=np.array(np.float32([0.0]*self.h3_length)); dbi3=np.array(np.float32([0.0]*self.h3_length))
            #forget gate:
            dWxf3=np.array(np.float32([[0.0]*self.io_length for n in range(self.h3_length)])); dWfj3=np.array(np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]))
            dWhf3=np.array(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)])); dWcf3=np.array(np.float32([0.0]*self.h3_length)); dbf3=np.array(np.float32([0.0]*self.h3_length))
            #output gate:
            dWxo3=np.array(np.float32([[0.0]*self.io_length for n in range(self.h3_length)])); dWoj3=np.array(np.float32([[0.0 for m in range(self.h2_length)] for n in range(self.h3_length)]))
            dWho3=np.array(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)])); dWco3=np.array(np.float32([0.0]*self.h3_length)); dbo3=np.array(np.float32([0.0]*self.h3_length))
            #state:
            dWxc3=np.array(np.float32([[0.0]*self.io_length for n in range(self.h3_length)])); dWcj3=np.array(np.float32([[0.0]*self.h2_length for n in range(self.h3_length)]))
            dWhc3=np.array(np.float32([[0.0]*self.h3_length for n in range(self.h3_length)])); dbc3=np.array(np.float32([0.0]*self.h3_length))
            ###### output layer RMSprop previous gradients ###### 
            dWhy1=np.array(np.float32([[0.0]*self.h1_length for n in range(self.io_length)])); dWhy2=np.array(np.float32([[0.0]*self.h2_length for n in range(self.io_length)]))
            dWhy3=np.array(np.float32([[0.0]*self.h3_length for n in range(self.io_length)])); dby=np.array(np.float32([0.0]*self.io_length))
           

            # each mini batch, compute gradients, take average gradient, then update weights.
            for n in np.arange(MBn):
                [Wxi1r, Wxi2r, Wxi3r, Wxf1r, Wxf2r, Wxf3r,
                 Wxo1r, Wxo2r, Wxo3r, Wxc1r, Wxc2r, Wxc3r,        
                 Whi1r, Whi2r, Whi3r, Whf1r, Whf2r, Whf3r,
                 Who1r, Who2r, Who3r, Whc1r, Whc2r, Whc3r,
                 Why1r, Why2r, Why3r,
                 Wxj2r, Wxj3r, Wfj2r, Wfj3r, Wcj2r, Wcj3r, Woj2r, Woj3r,        
                 Wci1r, Wci2r, Wci3r, Wcf1r, Wcf2r, Wcf3r, Wco1r, Wco2r, Wco3r,
                 bi1r, bi2r, bi3, bf1r, bf2r, bf3r, bc1r, bc2r, bc3r, bo1r, bo2r, bo3r, byr,
                 Th1AccR, Th2AccR, Th3AccR, Tc1AccR, Tc2AccR, Tc3AccR,
                 TytAccR, Th1AccR, D_YtAccR, D_It_3AccR, D_It_2AccR, D_It_1AccR, D_Ft_3AccR, D_Ft_2AccR, D_Ft_1AccR, D_Ct_3AccR, D_Ct_2AccR, D_Ct_1AccR, D_Ot_3AccR, D_Ot_2AccR, D_Ot_1AccR, 
                 D_ct_3AccR, D_ct_2AccR, D_ct_1AccR, D_ot_3AccR, D_ot_2AccR, D_ot_1AccR, D_ft_3AccR, D_ft_2AccR, D_ft_1AccR, D_it_3AccR, D_it_2AccR, D_it_1AccR, 
                    KLAccR] = singleGradfn(np.float32(np.array(mbDataset)[n,0:tuneLength-2,:]), np.float32(np.array(mbDataset)[n,1:tuneLength-1,:]), \
                                            np.float32(np.asarray([0.0]*self.h3_length)), np.float32(np.asarray([0.0]*self.h2_length)), np.float32(np.asarray([0.0]*self.h1_length)), \
                                            np.float32(np.asarray([0.0]*self.h3_length)), np.float32(np.asarray([0.0]*self.h2_length)), np.float32(np.asarray([0.0]*self.h1_length)), \
                                            np.float32(np.asarray([0.0]*self.h3_length)), np.float32(np.asarray([0.0]*self.h2_length)), np.float32(np.asarray([0.0]*self.h1_length)), \
                                            np.float32(np.asarray([0.0]*self.h3_length)), np.float32(np.asarray([0.0]*self.h2_length)), np.float32(np.asarray([0.0]*self.h1_length)) )
    

                dWxi1 += Wxi1r/MBn; dWxi2 += Wxi2r/MBn; dWxi3 += Wxi3r/MBn; dWxf1 += Wxf1r/MBn; dWxf2 += Wxf2r/MBn; dWxf3 += Wxf3r/MBn
                dWxo1 += Wxo1r/MBn; dWxo2 += Wxo2r/MBn; dWxo3 += Wxo3r/MBn; dWxc1 += Wxc1r/MBn; dWxc2 += Wxc2r/MBn; dWxc3 += Wxc3r/MBn        
                dWhi1 += Whi1r/MBn; dWhi2 += Whi2r/MBn; dWhi3 += Whi3r/MBn; dWhf1 += Whf1r/MBn; dWhf2 += Whf2r/MBn; dWhf3 += Whf3r/MBn
                dWho1 += Who1r/MBn; dWho2 += Who2r/MBn; dWho3 += Who3r/MBn; dWhc1 += Whc1r/MBn; dWhc2 += Whc2r/MBn; dWhc3 += Whc3r/MBn
                dWhy1 += Why1r/MBn; dWhy2 += Why2r/MBn; dWhy3 += Why3r/MBn
                dWxj2 += Wxj2r/MBn; dWxj3 += Wxj3r/MBn; dWfj2 += Wfj2r/MBn; dWfj3 += Wfj3r/MBn; dWcj2 += Wcj2r/MBn; dWcj3 += Wcj3r/MBn; dWoj2 += Woj2r/MBn; dWoj3 += Woj3r/MBn        
                dWci1 += Wci1r/MBn; dWci2 += Wci2r/MBn; dWci3 += Wci3r/MBn; dWcf1 += Wcf1r/MBn; dWcf2 += Wcf2r/MBn; dWcf3 += Wcf3r/MBn; dWco1 += Wco1r/MBn; dWco2 += Wco2r/MBn; dWco3 += Wco3r/MBn
                dbi1 += bi1r/MBn; dbi2 += bi2r/MBn; dbi3 += bi3/MBn; dbf1 += bf1r/MBn; dbf2 += bf2r/MBn; dbf3 += bf3r/MBn 
                dbc1 += bc1r/MBn; dbc2 += bc2r/MBn; dbc3 += bc3r/MBn; dbo1 += bo1r/MBn; dbo2 += bo2r/MBn; dbo3 += bo3r/MBn; dby += byr/MBn
                epochLoss += KLAccR/MBn
            #update weights with averaged gradient in the MB:
            self.updateWeights(dWxi1, dWxi2, dWxi3, dWxf1, dWxf2, dWxf3, dWxo1, dWxo2, dWxo3, dWxc1, dWxc2, dWxc3, dWhi1, dWhi2, dWhi3, 
                                    dWhf1, dWhf2, dWhf3, dWho1, dWho2, dWho3, dWhc1, dWhc2, dWhc3, dWhy1, dWhy2, dWhy3,
                                    dWxj2, dWxj3, dWfj2, dWfj3, dWcj2, dWcj3, dWoj2, dWoj3, dWci1, dWci2, dWci3, dWcf1, dWcf2, dWcf3, 
                                    dWco1, dWco2, dWco3, dbi1, dbi2, dbi3, dbf1, dbf2, dbf3, dbc1, dbc2, dbc3, dbo1, dbo2, dbo3, dby)
            print("time taken = " + str(time.time()-pretime) + ", loss = " + str(epochLoss) + ", epoch " + str(m))

   

                
    def updateWeights(self, dWxi1, dWxi2, dWxi3, dWxf1, dWxf2, dWxf3, dWxo1, dWxo2, dWxo3, dWxc1, dWxc2, dWxc3, dWhi1, dWhi2, dWhi3, dWhf1, dWhf2, dWhf3, dWho1, dWho2, dWho3, dWhc1, dWhc2, dWhc3, dWhy1, dWhy2, dWhy3,
                            dWxj2, dWxj3, dWfj2, dWfj3, dWcj2, dWcj3, dWoj2, dWoj3, dWci1, dWci2, dWci3, dWcf1, dWcf2, dWcf3, dWco1, dWco2, dWco3, dbi1, dbi2, dbi3, dbf1, dbf2, dbf3, dbc1, dbc2, dbc3, dbo1, dbo2, dbo3, dby):
        #RMS prop update gradient of all weights with previously saved gradient and newly averaged (over MB) gradient. newly averaged gradients have magnitudes clipped to self.gradClip  
        self.DWxi1p.set_value(self.RMSgrad_np(self.DWxi1p.eval(), self.gClip_np(dWxi1)));  self.DWxi2p.set_value(self.RMSgrad_np(self.DWxi2p.eval(), self.gClip_np(dWxi2)));  self.DWxi3p.set_value(self.RMSgrad_np(self.DWxi3p.eval(), self.gClip_np(dWxi3)))
        self.DWxf1p.set_value(self.RMSgrad_np(self.DWxf1p.eval(), self.gClip_np(dWxf1)));  self.DWxf2p.set_value(self.RMSgrad_np(self.DWxf2p.eval(), self.gClip_np(dWxf2)));  self.DWxf3p.set_value(self.RMSgrad_np(self.DWxf3p.eval(), self.gClip_np(dWxf3)))
        self.DWxo1p.set_value(self.RMSgrad_np(self.DWxo1p.eval(), self.gClip_np(dWxo1)));  self.DWxo2p.set_value(self.RMSgrad_np(self.DWxo2p.eval(), self.gClip_np(dWxo2)));  self.DWxo3p.set_value(self.RMSgrad_np(self.DWxo3p.eval(), self.gClip_np(dWxo3)))
        self.DWxc1p.set_value(self.RMSgrad_np(self.DWxc1p.eval(), self.gClip_np(dWxc1)));  self.DWxc2p.set_value(self.RMSgrad_np(self.DWxc2p.eval(), self.gClip_np(dWxc2)));  self.DWxc3p.set_value(self.RMSgrad_np(self.DWxc3p.eval(), self.gClip_np(dWxc3)))
        
        self.DWhi1p.set_value(self.RMSgrad_np(self.DWhi1p.eval(), self.gClip_np(dWhi1)));  self.DWhi2p.set_value(self.RMSgrad_np(self.DWhi2p.eval(), self.gClip_np(dWhi2)));  self.DWhi3p.set_value(self.RMSgrad_np(self.DWhi3p.eval(), self.gClip_np(dWhi3)))
        self.DWhf1p.set_value(self.RMSgrad_np(self.DWhf1p.eval(), self.gClip_np(dWhf1)));  self.DWhf2p.set_value(self.RMSgrad_np(self.DWhf2p.eval(), self.gClip_np(dWhf2)));  self.DWhf3p.set_value(self.RMSgrad_np(self.DWhf3p.eval(), self.gClip_np(dWhf3)))
        self.DWho1p.set_value(self.RMSgrad_np(self.DWho1p.eval(), self.gClip_np(dWho1)));  self.DWho2p.set_value(self.RMSgrad_np(self.DWho2p.eval(), self.gClip_np(dWho2)));  self.DWho3p.set_value(self.RMSgrad_np(self.DWho3p.eval(), self.gClip_np(dWho3)))
        self.DWhc1p.set_value(self.RMSgrad_np(self.DWhc1p.eval(), self.gClip_np(dWhc1)));  self.DWhc2p.set_value(self.RMSgrad_np(self.DWhc2p.eval(), self.gClip_np(dWhc2)));  self.DWhc3p.set_value(self.RMSgrad_np(self.DWhc3p.eval(), self.gClip_np(dWhc3)))
        
        self.DWhy1p.set_value(self.RMSgrad_np(self.DWhy1p.eval(), self.gClip_np(dWhy1)));  self.DWhy2p.set_value(self.RMSgrad_np(self.DWhy2p.eval(), self.gClip_np(dWhy2)));  self.DWhy3p.set_value(self.RMSgrad_np(self.DWhy3p.eval(), self.gClip_np(dWhy3)))

        self.DWxj2p.set_value(self.RMSgrad_np(self.DWxj2p.eval(), self.gClip_np(dWxj2)));  self.DWxj3p.set_value(self.RMSgrad_np(self.DWxj3p.eval(), self.gClip_np(dWxj3)))
        self.DWfj2p.set_value(self.RMSgrad_np(self.DWfj2p.eval(), self.gClip_np(dWfj2)));  self.DWfj3p.set_value(self.RMSgrad_np(self.DWfj3p.eval(), self.gClip_np(dWfj3)))
        self.DWcj2p.set_value(self.RMSgrad_np(self.DWcj2p.eval(), self.gClip_np(dWcj2)));  self.DWcj3p.set_value(self.RMSgrad_np(self.DWcj3p.eval(), self.gClip_np(dWcj3)))
        self.DWoj2p.set_value(self.RMSgrad_np(self.DWoj2p.eval(), self.gClip_np(dWoj2)));  self.DWoj3p.set_value(self.RMSgrad_np(self.DWoj3p.eval(), self.gClip_np(dWoj3)))
        
        self.DWci1p.set_value(self.RMSgrad_np(self.DWci1p.eval(), self.gClip_np(dWci1)));  self.DWci2p.set_value(self.RMSgrad_np(self.DWci2p.eval(), self.gClip_np(dWci2)));  self.DWci3p.set_value(self.RMSgrad_np(self.DWci3p.eval(), self.gClip_np(dWci3)))
        self.DWcf1p.set_value(self.RMSgrad_np(self.DWcf1p.eval(), self.gClip_np(dWcf1)));  self.DWcf2p.set_value(self.RMSgrad_np(self.DWcf2p.eval(), self.gClip_np(dWcf2)));  self.DWcf3p.set_value(self.RMSgrad_np(self.DWcf3p.eval(), self.gClip_np(dWcf3)))
        self.DWco1p.set_value(self.RMSgrad_np(self.DWco1p.eval(), self.gClip_np(dWco1)));  self.DWco2p.set_value(self.RMSgrad_np(self.DWco2p.eval(), self.gClip_np(dWco2)));  self.DWco3p.set_value(self.RMSgrad_np(self.DWco3p.eval(), self.gClip_np(dWco3)))
        
        self.Dbi1p.set_value(self.RMSgrad_np(self.Dbi1p.eval(), self.gClip_np(dbi1))); self.Dbi2p.set_value(self.RMSgrad_np(self.Dbi2p.eval(), self.gClip_np(dbi2))); self.Dbi3p.set_value(self.RMSgrad_np(self.Dbi3p.eval(), self.gClip_np(dbi3)))
        self.Dbf1p.set_value(self.RMSgrad_np(self.Dbf1p.eval(), self.gClip_np(dbf1))); self.Dbf2p.set_value(self.RMSgrad_np(self.Dbf2p.eval(), self.gClip_np(dbf2))); self.Dbf3p.set_value(self.RMSgrad_np(self.Dbf3p.eval(), self.gClip_np(dbf3)))
        self.Dbc1p.set_value(self.RMSgrad_np(self.Dbc1p.eval(), self.gClip_np(dbc1))); self.Dbc2p.set_value(self.RMSgrad_np(self.Dbc2p.eval(), self.gClip_np(dbc2))); self.Dbc3p.set_value(self.RMSgrad_np(self.Dbc3p.eval(), self.gClip_np(dbc3)))
        self.Dbo1p.set_value(self.RMSgrad_np(self.Dbo1p.eval(), self.gClip_np(dbo1))); self.Dbo2p.set_value(self.RMSgrad_np(self.Dbo2p.eval(), self.gClip_np(dbo2))); self.Dbo3p.set_value(self.RMSgrad_np(self.Dbo3p.eval(), self.gClip_np(dbo3)))
        self.Dbyp.set_value(self.RMSgrad_np(self.Dbyp.eval(), self.gClip_np(dby)))      

        self.Wxi_1.set_value(self.Wxi_1.eval()-self.R1 *self.DWxi1p.eval());  self.Wxi_2.set_value(self.Wxi_2.eval()-self.R2 *self.DWxi2p.eval());  self.Wxi_3.set_value(self.Wxi_3.eval()-self.R3 *self.DWxi3p.eval())
        self.Wxf_1.set_value(self.Wxf_1.eval()-self.R1 *self.DWxf1p.eval());  self.Wxf_2.set_value(self.Wxf_2.eval()-self.R2 *self.DWxf2p.eval());  self.Wxf_3.set_value(self.Wxf_3.eval()-self.R3 *self.DWxf3p.eval())
        self.Wxo_1.set_value(self.Wxo_1.eval()-self.R1 *self.DWxo1p.eval());  self.Wxo_2.set_value(self.Wxo_2.eval()-self.R2 *self.DWxo2p.eval());  self.Wxo_3.set_value(self.Wxo_3.eval()-self.R3 *self.DWxo3p.eval())
        self.Wxc_1.set_value(self.Wxc_1.eval()-self.R1 *self.DWxc1p.eval());  self.Wxc_2.set_value(self.Wxc_2.eval()-self.R2 *self.DWxc2p.eval());  self.Wxc_3.set_value(self.Wxc_3.eval()-self.R3 *self.DWxc3p.eval())
        
        self.Whi_1.set_value(self.Whi_1.eval()-self.R1 *self.DWhi1p.eval());  self.Whi_2.set_value(self.Whi_2.eval()-self.R2 *self.DWhi2p.eval());  self.Whi_3.set_value(self.Whi_3.eval()-self.R3 *self.DWhi3p.eval())
        self.Whf_1.set_value(self.Whf_1.eval()-self.R1 *self.DWhf1p.eval());  self.Whf_2.set_value(self.Whf_2.eval()-self.R2 *self.DWhf2p.eval());  self.Whf_3.set_value(self.Whf_3.eval()-self.R3 *self.DWhf3p.eval())
        self.Who_1.set_value(self.Who_1.eval()-self.R1 *self.DWho1p.eval());  self.Who_2.set_value(self.Who_2.eval()-self.R2 *self.DWho2p.eval());  self.Who_3.set_value(self.Who_3.eval()-self.R3 *self.DWho3p.eval())
        self.Whc_1.set_value(self.Whc_2.eval()-self.R1 *self.DWhc1p.eval());  self.Whc_2.set_value(self.Whc_2.eval()-self.R2 *self.DWhc2p.eval());  self.Whc_3.set_value(self.Whc_3.eval()-self.R3 *self.DWhc3p.eval())
        
        self.Why_1.set_value(self.Why_1.eval()-self.R1 *self.DWhy1p.eval());  self.Why_2.set_value(self.Why_2.eval()- self.R2 *self.DWhy2p.eval());  self.Why_3.set_value(self.Why_3.eval()-self.R3 *self.DWhy3p.eval())

        self.Wxj_2.set_value(self.Wxj_2.eval()-self.R2 *self.DWxj2p.eval());  self.Wxj_3.set_value(self.Wxj_3.eval()-self.R3 *self.DWxj3p.eval())
        self.Wfj_2.set_value(self.Wfj_2.eval()-self.R2 *self.DWfj2p.eval());  self.Wfj_3.set_value(self.Wfj_3.eval()-self.R3 *self.DWfj3p.eval())
        self.Wcj_2.set_value(self.Wcj_2.eval()-self.R2 *self.DWcj2p.eval());  self.Wcj_3.set_value(self.Wcj_3.eval()-self.R3 *self.DWcj3p.eval())
        self.Woj_2.set_value(self.Woj_2.eval()-self.R2 *self.DWoj2p.eval());  self.Woj_3.set_value(self.Woj_3.eval()-self.R3 *self.DWoj3p.eval())
        
        self.Wci_1.set_value(self.Wci_1.eval()-self.R1 *self.DWci1p.eval());  self.Wci_2.set_value(self.Wci_2.eval()-self.R2 *self.DWci2p.eval());  self.Wci_3.set_value(self.Wci_3.eval()-self.R3 *self.DWci3p.eval())
        self.Wcf_1.set_value(self.Wcf_1.eval()-self.R1 *self.DWcf1p.eval());  self.Wcf_2.set_value(self.Wcf_2.eval()-self.R2 *self.DWcf2p.eval());  self.Wcf_3.set_value(self.Wcf_3.eval()-self.R3 *self.DWcf3p.eval())
        self.Wco_1.set_value(self.Wco_1.eval()-self.R1 *self.DWco1p.eval());  self.Wco_2.set_value(self.Wco_2.eval()-self.R2 *self.DWco2p.eval());  self.Wco_3.set_value(self.Wco_3.eval()-self.R3 *self.DWco3p.eval())
        
        self.bi_1.set_value(self.bi_1.eval()-self.R1 *self.Dbi1p.eval()); self.bi_2.set_value(self.bi_2.eval()-self.R2 *self.Dbi2p.eval()); self.bi_3.set_value(self.bi_3.eval()-self.R3 *self.Dbi3p.eval())
        self.bf_1.set_value(self.bf_1.eval()-self.R1 *self.Dbf1p.eval()); self.bf_2.set_value(self.bf_2.eval()-self.R2 *self.Dbf2p.eval()); self.bf_3.set_value(self.bf_3.eval()-self.R3 *self.Dbf3p.eval())
        self.bc_1.set_value(self.bc_1.eval()-self.R1 *self.Dbc1p.eval()); self.bc_2.set_value(self.bc_2.eval()-self.R2 *self.Dbc2p.eval()); self.bc_3.set_value(self.bc_3.eval()-self.R3 *self.Dbc3p.eval())
        self.bo_1.set_value(self.bo_1.eval()-self.R1 *self.Dbo1p.eval()); self.bo_2.set_value(self.bo_2.eval()-self.R2 *self.Dbo2p.eval()); self.bo_3.set_value(self.bo_3.eval()-self.R3 *self.Dbo3p.eval())
        self.by.set_value(self.by.eval()-self.Rout *self.Dbyp.eval())
    


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
    

        
def main():
    sizeOfMiniBatch = 5
    noOfEpoch = 100
    path = './Piano-midi.de/train-individual/hpps'
    files = os.listdir(path)
    assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
    #pitch range is from 21 to 109
    dataset = [midiread((path + "/" + f), (21, 109),0.3).piano_roll.astype(theano.config.floatX) for f in files]
    #to check size, use print(str(np.array(dataset).shape)) and print(str(np.array(dataset[data#]).shape))
    #np.transpose(dataset) gives you dataset[data stream#, 0-87][sample, 0-575][onehot encoded in a 88 sized vector, 0-87]
    #print("np.array(dataset).shape = " + str(np.array(dataset).shape))
    #print("np.array(dataset[1]).shape = " + str(np.array(dataset[1]).shape))
    #print("np.array(dataset[40]).shape = " + str(np.array(dataset[40]).shape))
    #print("np.array(dataset[40]).shape[0] = " + str(np.array(dataset[40]).shape[0]))
    #print("np.transpose(np.array(dataset[1])).shape = " + str(np.transpose(np.array(dataset[1])).shape))
  
    #print(str(dataset[40][400][87]))
    #print(str([dataset[40][n][87] for n in np.arange(0,601,1)])) #np.arnage(start, stop, step), dataset[40][400:405][87] does not work like matlab does... 
    #LSTM crashes when input have "blanks" i.e. non of the 88 possible notes are "1.0"
    #we detect these blank notes and represent it with note #0 :
    
    #plt.show()
    
    
    for n in np.arange(0,np.array(dataset[3]).shape[0],1):
        if np.sum(dataset[3][n], dtype=theano.config.floatX) == 0 :
            dataset[3][n][0] = np.float32(1.0)
    for n in np.arange(0,np.array(dataset[8]).shape[0],1):
        if np.sum(dataset[8][n], dtype=theano.config.floatX) == 0 :
            dataset[8][n][0] = np.float32(1.0)
                
    print("maximum of dataset[3] = " + str(np.max(dataset[3])))
    print("minimum of dataset[3] = " + str(np.min(dataset[3])))
  
    #shifting input to zero centre does not work as crossing zero blows things up...


    #check number of notes for each tune:
       
    print(str([np.array(dataset[n]).shape[0] for n in np.arange(np.array(dataset).shape[0])]))

    dataLength = [np.array(dataset[n]).shape[0] for n in np.arange(np.array(dataset).shape[0])]

    # set "silent" to zero in 1-hot format
    for k in np.arange(np.array(dataset).shape[0]):
        for n in np.arange(0,np.array(dataset[k]).shape[0],1):
            if np.sum(dataset[k][n], dtype=theano.config.floatX) == 0 :
                dataset[k][n][0] = np.float32(1.0)
                
    ## set data to +/-1
    #for k in np.arange(np.array(dataset).shape[0]):
    #    for n in np.arange(0,np.array(dataset[k]).shape[0],1):
    #        dataset[k][n][0] = np.float32(2.0)*cp.deepcopy(dataset[k][n][0]) - np.float32(1.0)

    myRNN4Music = RNN4Music(h1_length=120, h2_length=120, h3_length=120, io_length=88, R1=0.01, R2=0.01, R3=0.01, Rout=0.01) #1000 (few million parameters)=>2.4s, 500=>0.8s, 2000=>9s, 2500=> out or ram! might need to clear and try again...
    #myRNN4Music.loadParameters('examples400_120') #this one works when you gen without softmax, all trained with* softmax though...strange... 
    #myRNN4Music.loadParameters('examples400_120_noSoftmax')
    #myRNN4Music.loadParameters('examples400_120_noSoftmax_RNN7_386plus')#('examples400_120_noSoftmax_RNN7')
    #myRNN4Music.loadParameters('examples400_120_noSoftmax_RNN7_596plus')
    #myRNN4Music.loadParameters('examples400_120_noSoftmax_RNN7_676plus')
    

    #myRNN4Music.loadParameters('fixedTrial_xEnError_109')    

    #examples400_120_noSoftmax_RNN7_676_r2plus sounds good on generating [100]
    #200 examples over 10 hrs
    for n in np.arange(0,np.array(dataset).shape[0],1): #np.arange(np.array(dataset).shape[0]):
         print('training with data[' + str(n) + ']')
         #myRNN4Music.resetStates()
         myRNN4Music.resetRMSgrads()   
         myRNN4Music.train(dataset, noOfEpoch, sizeOfMiniBatch, 120) 
         myRNN4Music.saveParameters('original')
         #myRNN4Music.printDerivatives()
         #myRNN4Music.printWeights()



#    myRNN4Music.saveParameters('fixedTrial')
#    myRNN4Music.resetStates()

## generate tunes with leading samples for all trained examples:
#    baseSample = 2
#    for baseSample in np.arange(0,np.array(dataset).shape[0],1):
#	myRNN4Music.resetStates()    
#	generated = myRNN4Music.genMusic(np.float32(dataset[baseSample][0:2]), 500)
#    	#print('generated: ' + str(np.array(generated)))
#    	print('generated tune ' + str(baseSample))
#	#plt.figure(0)
#    	#plt.imshow(np.transpose(dataset[baseSample][0:dataLength[baseSample]]), origin='lower', aspect='auto',
#        #                     interpolation='nearest', cmap=pylab.cm.gray_r)
#    
#    	#plt.figure(1)
#    	#plt.imshow(np.transpose(np.array(generated[0][0:dataLength[baseSample]])), origin='lower', aspect='auto',
#        #                     interpolation='nearest', cmap=pylab.cm.gray_r)
#    	#plt.show()
#    	
#    


    baseSample = 1
    exampleLength = 50
    myRNN4Music.resetStates()
    generatedTuneProb = myRNN4Music.genMusic(np.float32(dataset[baseSample][0:exampleLength]), 2000)
    midiwrite('fixedtrial_' + str(baseSample) + '182xEnError.mid', generatedTuneProb[0], (21, 109),0.3)
    #generatedTuneProb[0] is the tune, generatedTuneProb[1] is the probability at each iteration
    plt.figure(0)
    plt.imshow(np.array(generatedTuneProb[1][0:20,25:65]), origin = 'lower', extent=[25,65,0,20], aspect=1,
                    interpolation = 'nearest', cmap='gist_stern_r')
    plt.title('probability of generated midi note piano-roll')
    plt.xlabel('midi note')
    plt.ylabel('sample number (time steps)')
    plt.colorbar()

    plt.figure(1)
    plt.imshow(np.transpose(dataset[baseSample][0:dataLength[baseSample]]), origin='lower', aspect='auto',
                             interpolation='nearest', cmap=pylab.cm.gray_r)
    plt.colorbar()
    plt.title('original piano-roll')
    plt.xlabel('sample number (time steps)')
    plt.ylabel('midi note')

    plt.figure(2)
    plt.imshow(np.transpose(np.array(generatedTuneProb[0][0:500])), origin='lower', aspect='auto',
                             interpolation='nearest', cmap=pylab.cm.gray_r)
    plt.colorbar()
    plt.title('generated piano-roll')
    plt.xlabel('sample number (time steps)')
    plt.ylabel('midi note')
    plt.show()


    
    #print(str(generatedTuneProb[1][0:5]))
    #myRNN4Music.printWeights()
    

    
        
        
if __name__ == "__main__":
    main()


