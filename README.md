# tenserflow_1
Train model to predict for some new value

Libraries used here are-
1- tenserflow
2- numpy

Here only 1 layer of neuron is used.

For every value of xs, it makes a guess answer for ys then optimizer checks the guess and the original answer and loss is calculated. This repeats until it reaches number of epochs we have given.

The large data given the more possiblilty of accurate answer.
 
numpy is used to store data and then pass it to model.fit() to train it.

model.pridict is used along with parenthesis '()' and the value inside parenthesis for which output is to be preducted.
