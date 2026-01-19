# Episode 1-introduction.md 

## Activation functions
Look at the following activation functions:

**A. Sigmoid activation function**
The sigmoid activation function is given by:
$$ f(x) = \frac{1}{1 + e^{-x}} $$

![](fig/01_sigmoid.svg){alt='Plot of the sigmoid function' width='70%' align='left'}
<br clear="all" />

**B. ReLU activation function**
The Rectified Linear Unit (ReLU) activation function is defined as:
$$ f(x) = \max(0, x) $$

This involves a simple comparison and maximum calculation, which are basic operations that are computationally inexpensive.
It is also simple to compute the gradient: 1 for positive inputs and 0 for negative inputs.

![](fig/01_relu.svg){alt='Plot of the ReLU function'  width='70%' align='left'}
<br clear="all" />

**C. Linear (or identity) activation function (output=input)**
The linear activation function is simply the identity function:
$$ f(x) = x $$

![](fig/01_identity_function.svg){alt='Plot of the Identity function'  width='70%' align='left'}
<br clear="all" />


Combine the following statements to the correct activation function:

1. This function enforces the activation of a neuron to be between 0 and 1
2. This function is useful in regression tasks when applied to an output neuron
3. This function is the most popular activation function in hidden layers, since it introduces non-linearity in a computationally efficient way.
4. This function is useful in classification tasks when applied to an output neuron
5. (optional) For positive values this function results in the same activations as the identity function.
6. (optional) This function is not differentiable at 0
7. (optional) This function is the default for Dense layers (search the Keras documentation!)

*Activation function plots by Laughsinthestocks - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=44920411,
https://commons.wikimedia.org/w/index.php?curid=44920600, https://commons.wikimedia.org/w/index.php?curid=44920533*


## Neural network calculations
.

#### 1. Calculate the output for one neuron
Suppose we have:

- Input: X = (0, 0.5, 1)
- Weights: W = (-1, -0.5, 0.5)
- Bias: b = 1
- Activation function _relu_: `f(x) = max(x, 0)`

What is the output of the neuron?

_Note: You can use whatever you like: brain only, pen&paper, Python, Excel..._

#### 2. (optional) Calculate outputs for a network

Have a look at the following network where:

* $X_1$ and $X_2$ denote the two inputs of the network.
* $h_1$ and $h_2$ denote the two neurons in the hidden layer. They both have ReLU activation functions.
* $h_1$ and $h_2$ denotes the output neuron. It has a ReLU activation function.
* The value on the arrows represent the weight associated to that input to the neuron.
* $b_i$ denotes the bias term of that specific neuron
![](fig/01_xor_exercise.png){alt='A diagram of a neural network with 2 inputs, 2 hidden layer neurons, and 1 output.' width='400px'}

a. Calculate the output of the network for the following combinations of inputs:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | ..|
| 0  | 1  | ..|
| 1  | 0  | ..|
| 1  | 1  | ..|

b. What logical problem does this network solve?


## Exercise: Loss function

#### 1. Compute the Mean Squared Error
One of the simplest loss functions is the Mean Squared Error. MSE = $\frac{1}{n} \Sigma_{i=1}^n({y}-\hat{y})^2$ .
It is the mean of all squared errors, where the error is the difference between the predicted and expected value.
In the following table, fill in the missing values in the 'squared error' column. What is the MSE loss
for the predictions on these 4 samples?

| **Prediction** | **Expected value** | **Squared error** |
|----------------|--------------------|-------------------|
| 1              | -1                 | 4                 |
| 2              | -1                 | ..                |
| 0              | 0                  | ..                |
| 3              | 2                  | ..                |
|                | **MSE:**           | ..                |

#### 2. (optional) Huber loss
A more complicated and less used loss function for regression is the [Huber loss](https://keras.io/api/losses/regression_losses/#huber-class).

Below you see the Huber loss (green, delta = 1) and Squared error loss (blue)
as a function of `y_true - y_pred`.

![](fig/01_huber_loss.png){alt='Line plot comparing squared error loss function with the Huber loss function where delta = 1, showing the cost of prediction error of both functions equal where y_true - y_pred is between -1 and 1, then rising linearly with the Huber loss function as y_true diverges further from y_pred, as opposed to expontentially for the squared error function.' width='400px'}

Which loss function is more sensitive to outliers?


## Deep Learning Problems Exercise

Which of the following would you apply deep learning to?

1. Recognising whether or not a picture contains a bird.
2. Calculating the median and interquartile range of a dataset.
3. Identifying MRI images of a rare disease when only one or two example images available for training.
4. Identifying people in pictures after being trained only on cats and dogs.
5. Translating English into French.


## Deep learning workflow exercise

Think about a problem you would like to use deep learning to solve.

1. What do you want a deep learning system to be able to tell you?
2. What data inputs and outputs will you have?
3. Do you think you will need to train the network or will a pre-trained network be suitable?
4. What data do you have to train with? What preparation will your data need? Consider both the data you are going to predict/classify from and the data you will use to train the network.


# Episode 2-keras.md 


## Pairplot

Take a look at the pairplot we created. Consider the following questions:

* Is there any class that is easily distinguishable from the others?
* Which combination of attributes shows the best separation for all 3 class labels at once?
* (optional) Create a similar pairplot, but with `hue="sex"`. Explain the patterns you see.
Which combination of features distinguishes the two sexes best?


## One-hot encoding
How many output neurons will our network have now that we one-hot encoded the target class?

* A: 1
* B: 2
* C: 3


## Create the neural network
With the code snippets above, we defined a Keras model with 1 hidden layer with
10 neurons and an output layer with 3 neurons.

1. How many parameters does the resulting model have?
2. What happens to the number of parameters if we increase or decrease the number of neurons
in the hidden layer?

#### (optional) Visualizing the model
Optionally, you can also visualize the same information as `model.summary()` in graph form.
This step requires the command-line tool `dot` from Graphviz installed, you installed it by following the setup instructions.
You can check that the installation was successful by executing `dot -V` in the command line. You should get something
as follows:

```sh
$ dot -V
dot - graphviz version 2.43.0 (0)
```

3. (optional) Provided you have `dot` installed, execute the `plot_model` function
as shown below.

```python
keras.utils.plot_model(
model,
show_shapes=True,
show_layer_names=True,
show_layer_activations=True,
show_trainable=True
)
```

#### (optional) Keras Sequential vs Functional API
So far we have used the [Functional API](https://keras.io/guides/functional_api/) of Keras.
You can also implement neural networks using [the Sequential model](https://keras.io/guides/sequential_model/).
As you can read in the documentation, the Sequential model is appropriate for **a plain stack of layers**
where each layer has **exactly one input tensor and one output tensor**.

4. (optional) Use the Sequential model to implement the same network


## The Training Curve
Looking at the training curve we have just made.

1. How does the training progress?
* Does the training loss increase or decrease?
* Does it change quickly or slowly?
* Does the graph look very jittery?
2. Do you think the resulting trained network will work well on the test set?

When the training process does not go well:

3. (optional) Something went wrong here during training. What could be the problem, and how do you see that in the training curve?
Also compare the range on the y-axis with the previous training curve.
![][bad-training-curve]


## Confusion Matrix
Measure the performance of the neural network you trained and
visualize a confusion matrix.

- Did the neural network perform well on the test set?
- Did you expect this from the training loss you saw?
- What could we do to improve the performance?


# Episode 3-monitor-the-model.md 

## Exercise: Architecture of the network
As we want to design a neural network architecture for a regression task,
see if you can first come up with the answers to the following questions:

1. What must be the dimension of our input layer?
2. We want to output the prediction of a single number. The output layer of the NN hence cannot be the same as for the classification task earlier. This is because the `softmax` activation being used had a concrete meaning with respect to the class labels which is not needed here. What output layer design would you choose for regression?
Hint: A layer with `relu` activation, with `sigmoid` activation or no activation at all?
3. (Optional) How would we change the model if we would like to output a prediction of the precipitation in Basel in *addition* to the sunshine hours?



## Exercise: Gradient descent

Answer the following questions:

### 1. What is the goal of optimization?

- A. To find the weights that maximize the loss function
- B. To find the weights that minimize the loss function

### 2. What happens in one gradient descent step?

- A. The weights are adjusted so that we move in the direction of the gradient, so up the slope of the loss function
- B. The weights are adjusted so that we move in the direction of the gradient, so down the slope of the loss function
- C. The weights are adjusted so that we move in the direction of the negative gradient, so up the slope of the loss function
- D. The weights are adjusted so that we move in the direction of the negative gradient, so down the slope of the loss function

### 3. When the batch size is increased:
(multiple answers might apply)

- A. The number of samples in an epoch also increases
- B. The number of batches in an epoch goes down
- C. The training progress is more jumpy, because more samples are consulted in each update step (one batch).
- D. The memory load (memory as in computer hardware) of the training process is increased


## Exercise: Reflecting on our results
* Is the performance of the model as you expected (or better/worse)?
* Is there a noteable difference between training set and test set? And if so, any idea why?
* (Optional) When developing a model, you will often vary different aspects of your model like
which features you use, model parameters and architecture. It is important to settle on a
single-number evaluation metric to compare your models.
* What single-number evaluation metric would you choose here and why?


## Exercise: Baseline
1. Looking at this baseline: Would you consider this a simple or a hard problem to solve?
2. (Optional) Can you think of other baselines?


## Exercise: plot the training progress.
1. Is there a difference between the training curves of training versus validation data? And if so, what would this imply?
2. (Optional) Take a pen and paper, draw the perfect training and validation curves.
(This may seem trivial, but it will trigger you to think about what you actually would like to see)


## Exercise: Try to reduce the degree of overfitting by lowering the number of parameters
We can keep the network architecture unchanged (2 dense layers + a one-node output layer) and only play with the number of nodes per layer.
Try to lower the number of nodes in one or both of the two dense layers and observe the changes to the training and validation losses.
If time is short: Suggestion is to run one network with only 10 and 5 nodes in the first and second layer.

1. Is it possible to get rid of overfitting this way?
2. Does the overall performance suffer or does it mostly stay the same?
3. (optional) How low can you go with the number of parameters without notable effect on the performance on the validation set?


## Exercise: Simplify the model and add data
You may have been wondering why we are including weather observations from
multiple cities to predict sunshine hours only in Basel. The weather is
a complex phenomenon with correlations over large distances and time scales,
but what happens if we limit ourselves to only one city?

1. Since we will be reducing the number of features quite significantly,
we could afford to include more data. Instead of using only 3 years, use
8 or 9 years!
2. Only use the features in the dataset that are for Basel, remove the data for other cities.
You can use something like:
```python
cols = [c for c in X_data.columns if c[:5] == 'BASEL']
X_data = X_data[cols]
```
3. Now rerun the last model we defined which included the BatchNorm layer.
Recreate the scatter plot comparing your predictions with the true values,
and evaluate the model by computing the RMSE on the test score.
Note that even though we will use many more observations than previously,
the network should still train quickly because we reduce the number of
features (columns).
Is the prediction better compared to what we had before?
4. (Optional) Try to train a model on all years that are available,
and all features from all cities. How does it perform?



## Open question: What could be next steps to further improve the model?

With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results.
Usually models are "well behaving" in the sense that small changes to the architectures also only result in small changes of the performance (if any).
It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist?
Applying common sense is often a good first step to make a guess of how much better results *could* be.
In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision.
But how much better our model could be exactly, often remains difficult to answer.

* What changes to the model architecture might make sense to explore?
* Ignoring changes to the model architecture, what might notably improve the prediction quality?


# Episode 4-advanced-layer-types.md 


### Number of features in Dollar Street 10

How many features does one image in the Dollar Street 10 dataset have?

- A. 64
- B. 4096
- C. 12288
- D. 878



## Number of parameters{#parameters-exercise-1}
Suppose we create a single Dense (fully connected) layer with 100 hidden units that connect to the input pixels, how many parameters does this layer have?

- A. 1228800
- B. 1228900
- C. 100
- D. 12288


## Border pixels
What, do you think, happens to the border pixels when applying a convolution?


## Number of model parameters{#parameters-exercise-3}
Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 64 * 64 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the answer of the earlier exercise, ["Number of Parameters"](#parameters-exercise-1).


## Convolutional Neural Network

Inspect the network above:

* What do you think is the function of the `Flatten` layer?
* Which layer has the most parameters? Do you find this intuitive?
* (optional) This dataset is similar to the often used CIFAR-10 dataset.
We can get inspiration for neural network architectures that could work on our dataset here: https://paperswithcode.com/sota/image-classification-on-cifar-10 . Pick a model and try to understand how it works.


## Network depth
What, do you think, will be the effect of adding a convolutional layer to your model? Will this model have more or fewer parameters?
Try it out. Create a `model` that has an additional `Conv2d` layer with 50 filters and another MaxPooling2D layer after the last MaxPooling2D layer. Train it for 10 epochs and plot the results.

**HINT**:
The model definition that we used previously needs to be adjusted as follows:
```python
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(50, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
# Add your extra layers here
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(50, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)
```


## Why and when to use convolutional neural networks
1. Would it make sense to train a convolutional neural network (CNN) on the penguins dataset and why?
2. Would it make sense to train a CNN on the weather dataset and why?
3. (Optional) Can you think of a different machine learning task that would benefit from a
CNN architecture?


## Vary dropout rate
1. What do you think would happen if you lower the dropout rate? Try it out, and
see how it affects the model training.
2. You are varying the dropout rate and checking its effect on the model performance,
what is the term associated to this procedure?



## Hyperparameter tuning

1: Looking at the grid search results, select all correct statements:

- A. 6 different models were trained in this grid search run, because there are 6 possible combinations for the defined hyperparameter values
- B. 2 different models were trained, 1 for each hyperparameter that we want to change
- C. 1 model is trained with 6 different hyperparameter combinations
- D. The model with 2 layer and a dropout rate of 0.5 is the best model with a validation loss of 2.144
- E. The model with 2 layers and a dropout rate of 0.8 is the best model with a validation loss of 2.086
- F. We found the model with the best possible combination of dropout rate and number of layers

2 (Optional): Perform a grid search finding the best combination of the following hyperparameters: 2 different activation functions: 'relu', and 'tanh', and 2 different values for the kernel size: 3 and 4. Which combination works best?

**Hint**: Instead of `hp.Int` you should now use `hp.Choice("name", ["value1", "value2"])` to use hyperparameters from a predefined set of possible values.


# Episode 5-transfer-learning.md 

## Inspect the DenseNet121 network
Have a look at the network architecture with `model.summary()`.
It is indeed a deep network, so expect a long summary!

### 1.Trainable parameters
How many parameters are there? How many of them are trainable?

Why is this and how does it effect the time it takes to train the model?

### 2. Head and base
Can you see in the model summary which part is the base network and which part is the head network?

### 3. Max pooling
Which layer is added because we provided `pooling='max'` as argument for `DenseNet121()`?


## Training and evaluating the pre-trained model

### 1. Compile the model
Compile the model:

- Use the `adam` optimizer
- Use the `SparseCategoricalCrossentropy` loss with `from_logits=True`.
- Use 'accuracy' as a metric.

### 2. Train the model
Train the model on the training dataset:

- Use a batch size of 32
- Train for 30 epochs, but use an earlystopper with a patience of 5
- Pass the validation dataset as validation data so we can monitor performance on the validation data during training
- Store the result of training in a variable called `history`
- Training can take a while, it is a much larger model than what we have seen so far.

### 3. Inspect the results
Plot the training history and evaluate the trained model. What do you think of the results?

### 4. (Optional) Try out other pre-trained neural networks
Train and evaluate another pre-trained model from https://keras.io/api/applications/. How does it compare to DenseNet121?



# Episode 6-outlook.md 

## Exercise: A real-world deep learning application

1. Looking at the 'Model training' section of the notebook, what do you recognize from what you learned in this course?
2. Can you identify the different steps of the deep learning workflow in this notebook?
3. (Optional): Try to understand the neural network architecture from the first figure of [the paper](https://doi.org/10.1186/s13321-021-00558-4).
a. Why are there 10.000 neurons in the input layer?
b. What do you think would happen if you would decrease the size of spectral embedding layer drastically, to for example 5 neurons?


