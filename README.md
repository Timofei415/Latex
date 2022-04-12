# Latex.py
The main project program file.

This project is a tool that can be used to write mathematical notes by transforming handwritten symbols (usually used in mathematics) into printed digits, letters, and other symbols.

The project is based on a neural network that can recognize 24 different symbols (the symbol list is provided below with handwritten examples. Since neural network was trained on my handwriting it will be able to recognize symbols if they are written in a similar way).

![symbollist](https://user-images.githubusercontent.com/71780377/162866648-e598dfb4-c811-460d-ac9a-16c7629616d6.png)


These are the most common symbols used in algebra for writing expressions, inequalities, derivatives, etc.

To start using the program press the New button creating an empty list. Then draw a symbol (preferably via notepad) the red frame will appear around the symbol and after that, it will be replaced by the image of the same symbol of the same size. You can draw symbols of any size anywhere on the canvas (convenient to denote powers, indices, etc.). In case the symbol is interpreted incorrectly you can remove it via undo button. 


# Neural_Network_training.py
This program is used to train a neural network that recognizes the symbols. 

The algorithm I used for this project is a neural network with one hidden layer. The program performs a backpropagation algorithm using the gradient descent method to minimize the cost. After training, the program saves the parameters theta1 and theta2 into the separate file parameters.py. The training is set with learning rate alpha = 1 and regression parameter lambda = 1. The choice of regression parameter was made via machine learning diagnostics. 

# diagnostics.py
This program performs the diagnostics of the neural network to find the best value for the regression parameter.

The program splits the dataset into a training set and cross-validation set in order to compare the error (cost divided by the number of examples) for parameters trained on the different number of training examples. The program plots a graph of training and cross-validation errors against the training examples to visualize the diagnostics. The graphs for lambda = 1 and lambda = 3 are also provided in this repository (diagnostics(reg=1).png and diagnostics(reg=3).png)

Note: the process of training neural network several times make take a while, so the diagnostics.py program takes 15-20 minutes to run.

![diagnostics(reg=1)](https://user-images.githubusercontent.com/71780377/162867299-cc0612cd-90c0-49df-a11a-457e7cdfcdf9.png)

![diagnostics(reg=3)](https://user-images.githubusercontent.com/71780377/162867325-5bdf5d5c-88dc-41c7-8ffa-73f1144bea98.png)






