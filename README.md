## LSTMmusic
Three layer Long-Short Term Memory network for music generation. References and more details can be found (here)[]. 

## setting up the environment
I developed this with Python 2.7 + [Theano](http://www.deeplearning.net/software/theano/install.html#install) on Ubuntu Linux 14.04. 

For the examples provided in LSTMmusic_main.py, an old i5 with 4GB ram runs more than smoothly. If you would like to experiment with larger number of units per layer with GPUs, Markus Beissinger has a great [page](http://markus.com/install-theano-on-aws/) on installing cuda on ubuntu (also covers Theano/Python if you are starting from scratch). 

Also download additional libraries to handle [midi](http://www.iro.umontreal.ca/~lisa/deep/midi.zip) files. Unzip these to `./midi` directory. The current directory `./` is the directory that includes all files in this repository. 

## example usage
The 3 Layer LSTM is implemented in the `RNN4music` class in LSTMmusicMB.py. A `main()` can be found as an example in LSTMmusic_main.py. To train the network with your own midi files, replace  

```
path= './Piano-midi./de/train-individual/... 
```

with the path containing all your midi training examples. The directory should only contain the .mid files but nothing else. 
In my case, the range of notes in the examples were between 21 to 109 (88 possible notes) with timestep set as 0.3:

```
dataset = [midiread((path + "/" + f), (21, 109),0.3).piano_roll.astype(theano.config.floatX) for f in files]
```

The code should load all training examples in `dataset`, having dimensions `[#example tunes, #length of each tune, #possible notes]`. So `dataset[1, 10, 33]` will indicate if note #33 at 10th time slot of example tune #1 should be played. 

you can set the size of each layers through the constructor of the `RNN4Music` class:

```
RNN4Music(h1_length=120, h2_length=120, h3_length=120, io_length=88, R1=np.float32(0.001), R2=np.float32(0.001), R3=np.float32(0.001), Rout=np.float32(0.001)) 
```

`io_length` is the number of possible notes in your midi files. In my case it was 88. Each layer (1, 2, 3, and output) can have its own training rate. Here I've set all to 0.001 in the example. Inside the constructor `RNN4Music()`, we can also set a few house keeping variables:

```
self.epsilon = np.float32(0.00001) 
```

Smallest number the code divides with. This guards against divide by zero errors.

```
self.gradClip = np.float32(30.0) 
```

this is the gradient upperbound during training. All weights/biases are initialised with Gaussian noise with the following standard deviation: 

```
self.sigRand = np.float32(0.1) 
```
all network variables (hidden states etc) at time -1 are initialised with uniform noise U[0,muRand] with parameter:

```
self.muRand = np.float32(0.01) 
```

As well as the network architecture and training rates, other training parameters such as `sizeOfMiniBatch`, `noOfEpoch` through the training set, and `noOfEpochPerMB` are set in the beginning of `main()`. The length of midi files in the training example path are not necessarily the same. For fairer training, the `lengthOfMB` parameter sets the length of each example tune feeding into the network during training. If a particular example is of length shorter than `lengthOfMB`, it will be filled up by repeating itself, but if the example have more samples than `lengthOfMB`, it will be truncated. Training is carried out by calling:

```train(dataset, noOfEpochPerMB, noOfEpoch, sizeOfMiniBatch, lengthOfMB)```

Two cost functions, sqaured error and cross-entropy are possible with the current code. To switch between the two, modify `backwardpass()` of the `RNN4music` class in LSTMmusicMB.py to comment out one of the following:

```
D_yt = - (kt / T.maximum(self.epsilon, yt)) + ((np.float32(1.0) - kt) / (1-T.minimum((np.float32(1.0) - self.epsilon),yt))) 
```
for cross-entropy cost, and:

```
D_yt = yt - kt 
```

for squared error cost.

A trained network can be saved and recalled using: 

```
saveParameters('file_name')
loadParameters('file_name')
```

These are instance methods so make sure the loaded network parameters fits the dimensions of the instantiated `RNN4music` object.

To generate music, use the method:

```
generatedTune = genMusic(np.float32(dataset[baseSample][0:exampleLength]), numberOfTimeSteps)
```

the first argument being a matrix of a short portion of an example tune to get the network started. I've used `exampleLength` here to specify a short example length (e.g. 20). After running through the example, the network self-sustains by feeding its own outputs as inputs for the next time step. The generated result is held in `generatedTune[0]`. Writing to a midi file can be carried out with the [midi package](http://www.iro.umontreal.ca/~lisa/deep/midi.zip):

```
midiwrite('file_name.mid', generatedTune[0], (21, 109),0.3)
```

the `(21,109)` specifies the possible note range and `0.3` is the midi time step. The probability of playing each note versus time can be accessed via the variable `generatedTune[1]`. We can also plot the note probabilities or notes themselves using `imshow()`:

```
plt.imshow(np.array(generatedTune[1][0:20,25:65]), origin = 'lower', extent=[25,65,0,20], aspect=1,
                                                      interpolation = 'nearest', cmap='gist_stern_r')
plt.imshow(np.transpose(np.array(generatedTune[0][0:500])), origin='lower', aspect='auto',
                                                      interpolation='nearest', cmap=pylab.cm.gray_r)
```

## License
Distributed by Henry Ip through the MIT License

