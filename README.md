In this implementation at first, we get the data related to the country's electricity as the input, divide the data into training and testing, then we have normalization pre-processing outliers. For modeling the LSTM network, we use Keras library and check the output with different network parameters to find the best parameters as a result of the best network for the given data. At last, we train the final network and perform the prediction of the test data and show the amount of training and test error in the form of a graph, and then we have the prediction of the next 24 hours with this designed network.
What we did in summary:
•	Preprocessing the data
•	Using LSTM deep learning model
•	Train and test loss for each epoch
•	Forecasting 24 hours ahead
•	Written in Python

Using google colab is the best choice to run this implementation.
