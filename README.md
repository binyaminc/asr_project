# asr_project
 
The project implemetns an automatic speech recognition system. We explore different NN architectures and data preprocessing.

## Data & Preprocessing 
The dataset we use is [AN4 dataset](https://drive.google.com/file/d/1MiPqJDm6gX_ayXZJ2LHeUbG0UNZfNagF/view?usp=sharing) from Carnegie Mellon University. It consists of people spelling out addresses, names, etc.


The input data is in .wav files. We convert the wav files into spectrograms (2D matrices of time-slices and frequencies). After that, we add a padding to the spectrograms, so that every input will have the same size. 

In order to create more data, in `add_data.py` we use 2 methods: add random noise to the wav files, and concatenating pairs of wav files to create new files. 

## Training
In `networks.py` we explored different CNN networks (in the future we want to explore RNN and transformers). The NN gets 1D wavs or 2D spectrograms, and the output is 2D matrix of character probability for each time slice. We used CTC loss. 

We started with 7 convolution layers, a linear layer and a log_softmax layer. We added Batch normalization, some more convolution layers, and then some more:). In addition, we checked diffenent additions of data in the preprocessing. 

We saved the trained networks with best results in `saved models` folder, so that we can load them later for prediction.
## Prediction
After we got the output matrix of char probabilities, we used beam search to find the approximation of the best "path" in the matrix (the best sequence of characters). Then we erased epsilon characters to get a final result.
