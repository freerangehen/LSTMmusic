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
import LSTMmusicMB
from LSTMmusicMB import RNN4Music


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

def main():

    #--- import data ---#
    sizeOfMiniBatch = 5 #how many tunes per miniBatch
    noOfEpoch = 100 
    noOfEpochPerMB = 2
    lengthOfMB = 100
    sparseParam = np.float32(0.01) #increases with no. of cells
    path = './Piano-midi.de/train-individual/hpps'
    #path = './Piano-midi.de/train'
    files = os.listdir(path)
    assert len(files) > 0, 'Training set is empty!' \
                               ' (did you download the data files?)'
    #pitch range is from 21 to 109
    dataset = [midiread((path + "/" + f), (21, 109),0.3).piano_roll.astype(theano.config.floatX) for f in files]
                  
    #check number of notes for each tune:       
    print(str([np.array(dataset[n]).shape[0] for n in np.arange(np.array(dataset).shape[0])]))

    # set "silent" to zero in 1-hot format
    for k in np.arange(np.array(dataset).shape[0]):
        for n in np.arange(0,np.array(dataset[k]).shape[0],1):
            if np.sum(dataset[k][n], dtype=theano.config.floatX) == 0 :
                dataset[k][n][0] = np.float32(1.0)
                

    #--- training with data ---#
    
    myRNN4Music = RNN4Music(h1_length=176, h2_length=176, h3_length=176, io_length=88, R1=np.float32(0.001), R2=np.float32(0.001), R3=np.float32(0.001), Rout=np.float32(0.001)) 
    
    #myRNN4Music.loadParameters('120_120_120_0_001_xEn_150epoch_hpps')
    #myRNN4Music.loadParameters('120_120_120_0_001_sqr_150epoch_hpps')
    myRNN4Music.loadParameters('176_176_176_0_001_xEn_L1_0_01_100epoch_hpps')
    #myRNN4Music.saveParameters('176_176_176_0_001_xEn_L1_0_1_300epoch_hpps')
    
    #myRNN4Music.train(dataset, noOfEpochPerMB, noOfEpoch, sizeOfMiniBatch, lengthOfMB, sparseParam)
    #myRNN4Music.saveParameters('176_176_176_0_001_xEn_L1_0_01_100epoch_hpps')
    #myRNN4Music.train(dataset, noOfEpochPerMB, noOfEpoch, sizeOfMiniBatch, lengthOfMB, sparseParam)
    #myRNN4Music.saveParameters('176_176_176_0_001_xEn_L1_0_01_200epoch_hpps')
    #myRNN4Music.train(dataset, noOfEpochPerMB, noOfEpoch, sizeOfMiniBatch, lengthOfMB, sparseParam)
    #myRNN4Music.saveParameters('176_176_176_0_001_xEn_L1_0_01_300epoch_hpps')
    #myRNN4Music.train(dataset, noOfEpochPerMB, noOfEpoch, sizeOfMiniBatch, lengthOfMB, sparseParam)
    #myRNN4Music.saveParameters('176_176_176_0_001_xEn_L1_0_01_400epoch_hpps')




    #--- plot some genearted tunes ---#

    for baseSample in np.array([0, 20, 15, 31]):
        exampleLength = 50
        myRNN4Music.resetStates()
        generatedTune = myRNN4Music.genMusic(np.float32(dataset[baseSample][0:exampleLength]), 300)
        midiwrite('176_176_176_0_001_sqr_hpps150_' + str(baseSample) + '.mid', generatedTune[0], (21, 109),0.3)
        #generatedTune[0] is the tune, generatedTune[1] is the probability at each iteration
        
        #plot genearted probability
        plt.figure(0 + baseSample*100)
        plt.imshow(np.array(generatedTune[1][0:50,25:65]), origin = 'lower', extent=[25,65,0,50], aspect=0.5,
                        interpolation = 'nearest', cmap='gist_stern_r')
        plt.title('probability of generated midi note piano-roll')
        plt.xlabel('midi note')
        plt.ylabel('sample number (time steps)')
        plt.colorbar()##
        #plot leading example for generation
        plt.figure(1 + baseSample*100)
        plt.imshow(np.transpose(dataset[baseSample]), origin='lower', aspect='auto',
                                 interpolation='nearest', cmap=pylab.cm.gray_r)
        plt.colorbar()
        plt.title('original piano-roll')
        plt.xlabel('sample number (time steps)')
        plt.ylabel('midi note')

        #plot generated tune
        plt.figure(2 + baseSample*100)
        plt.imshow(np.transpose(np.array(generatedTune[0][0:500])), origin='lower', aspect='auto',
                                 interpolation='nearest', cmap=pylab.cm.gray_r)
        plt.colorbar()
        plt.title('generated piano-roll')
        plt.xlabel('sample number (time steps)')
        plt.ylabel('midi note')
    plt.show()
    

    
        
        
if __name__ == "__main__":
    main()


