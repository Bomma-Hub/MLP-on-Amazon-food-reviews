# MLP-on-Amazon-food-reviews
Implementation of various MLP architectures on Amazon food reviews

Deep Learning

Deep Learning Training Flow:
1.	Load the dataset.
2.	Explore the data.
3.	Preprocess data.
4.	Build the Model.
a.	Set-up the Layers.
5.	Compile the model.
a.	Select Loss Function.
b.	Select Optimizer.
c.	Metrics to evaluate.
6.	Train the model.
7.	Evaluate Accuracy.
8.	Test the Model.
Neuron/Perceptron: 
1.	A single unit that has one node takes input and does some mathematical process gives output.
2.	Building block of Neural Network.
3.	A node is made up of activation function which will be activated when certain threshold value is given as Input.
4.	A single Neuron is also known as Perceptron.
5.	Perceptron is developed in around 1957
                                      
Note:
•	If multiple neurons stacked/placed one after other it becomes an Artificial Neural Network.
•	If the depth of neural network is Large then the Network becomes Deep Neural networks.
Logistic Regression as Perceptron: 
                 
•	If we use activation function (node) in perceptron as sigmoid then it behaves  similar to Logistic regression.
Notation:
 


Training a Single Neuron:
Step-1: Define the Loss function (Lr. Loss or Log. Loss etc …)
a.	If Loss is Linear ( Lr.) then the activation function is Identity function.
b.	If Loss is Log then the activation function is sigmoid function.
c.	If activation function Logistic function then the task we are handling is Classification.
d.	If activation function is Identity then the task we are handling is Regression.
                                      
Step-2: select the optimization problem.
                       


Step-3: Solve the optimization problem.
a.	Initialize weights.
b.	Find the partial derivative of weight vector.
    

Training a Deep Neuron Model:
The Step for training deep leaning model will follow same pattern as of single neuron model.
Step-1: Propose the optimization problem i..e Loss function.
Step-2: Select optimization technique to get the minimized loss eight vector. i.e like GD(Gradient descent).SGD (stochastic gradient Descent) etc.
a.	Weight Initialization.
b.	Calculate new weights.
c.	Update new weights till converge              

              
                       
Memoization:
•	Compute once and re-use.
•	If there is any operation that is used multiple time repeatedly then it is good practice to compute it once, store and re-use.
•	It will speed up the processing but uses extra memory.
                       
BackPropogation:
Backprop = Chain Rule + Memoization.
1.	Back-propogation works iff activation function is differentiable.
2.	 
3.	 

Derivative is used to know how fast a function changes or rate of change.
                    
Activation Functions:
Sigmoid: 
1.	It is the most popular and widely used.
Refer: http://ronny.rest/blog/post_2017_08_10_sigmoid/ Refer:http://ronny.rest/blog/post_2017_08_16_tanh/
2.	It also fits well for Logistic regression.
3.	We can represent the derivative of sigmoid in sigmoid itself.
                              
Tanh:
                                               

                                               
Two most widely used activation functions because
1.	They are differentiable.(if it is not then back-propogation doesn’t work).
2.	Easy to differentiate.
Vanishing gradients:
One of the important reasons why neural Networks was sidelined during 80’s,90’s etc.
1.	As derivative of a function ranges from 0 to 1 and if we use sigmoid/tanh as activation function then to update the weights in a deep neural network (multiplication of derivatives i.e chain rule) will be in negligible value. 
2.	IF we run for few epoch’s then the values of derivatives will be almost ‘0’ and new weights are almost similar to new weights.
                            
                            

Drop-out Layers:
 
 
RELU Activation Function: (Rectified Linear Units)
This is used as optional activation function for sigmoid or tanh due to exploding gradient
This converges faster than classical activation functions
 
RELU function:
                                        
 

•	As we are aware of exploding and vanishing gradient problem of using sigmoid and tanh as activation functions, if we use RELU then we may face dead activations due to its derivative values ranges between 0 or 1.
•	i.e  if one of the derivative is zero (when all input weights are high negative values then W.T*X  will be negative and derivative value is zero), due to chain rule, total  gradient value becomes zero then new weights are equal to old weights which should not be the case.
•	If more activation functions are in dead state then the model is not learning(not updating to new weights) for this we use Leaky RELU.


Stochastic Gradient Descent + Momentum:
•	Stochastic gradient is used for optimization.
•	When it is used, the weights are updated with high noise.
•	To de-noise the weighted values we use SGD with respect to momentum.
•	For this we use exponential weighted gradient sums. In this we give more imp to recent weights and less importance to past weights. BY this the gradient converges faster than to be expected else it delays in convergence.
•	Refer to below link for more details.
Reference: 
•	http://ruder.io/optimizing-gradient-descent/.
•	https://www.slideshare.net/SebastianRuder/optimization-for-deep-learning.
Nesterov Accelerated Gradient:
Unlike SGD + Momentum, In this we first calculate momentum and then find the gradient term which makes the gradient to converge even more faster. Where as we calculate SGD + momentum to get new weights.
Reference for gradient optimizers: https://deepnotes.io/sgd-momentum-adaptive
ADAGRAD: ADAPTIVE GRADIENT:  
In this we use different learning rate in each epoch. 
Why? Due to sparse features.
AdaGrad (original paper) keeps track of per parameter sum of squared gradient and normalizes parameter update step. The idea is that parameters which receive big updates will have their effective learning rate reduced, while parameters which receive small updates will have their effective learning rate increased. This way we can accelerate the convergence by accelerating per parameter learning

A disadvantage of ADAGRAD is that the part of learning rate has monotonically increasing function which can steadily decrease the point where it stops the learning together

ADAM: Adaptive Momentum Estimation
Here we not only get the exponentially weighted sums of gradients but also the momentums as well by this.
combining best of the both momentum and adaptive learning worlds


SoftMax Classifier:
Logistic + Multiclass = SoftMax Classifier
 














KERAS

•	In Keras ,We will assemble layers to build ,models.
•	 Keras has three backend implementations available: the TensorFlow backend, the Theano backend, and the CNTK backend.
•	There are two main types of models available in Keras: the Sequential model, and the Model class used with the functional API.
•	Most common type of model is stack of layers : Sequential Model
https://towardsdatascience.com/building-a-deep-learning-model-using-keras-1548ca149d37
Sequential Model:
Input Layer:
`Input()` is used to instantiate a Keras tensor.

  A Keras tensor is a tensor object from the underlying backend  (Theano or TensorFlow), which we augment with certain  attributes that allow us to build a Keras model  just by knowing the inputs and outputs of the model.

  For instance, if a, b and c are Keras tensors,  it becomes possible to do:
  `model = Model(input=[a, b], output=c)`

  The added Keras attribute is:
      `_keras_history`: Last layer applied to the tensor. the entire layer graph is retrievable from that layer,
          recursively.         

  Returns:
    A `tensor`.
  Example:
  ```python
  # this is a logistic regression in Keras
  x = Input(shape=(32,))
  y = Dense(16, activation='softmax')(x)
  model = Model(x, y)
  ```

  Note that even if eager execution is enabled,
  `Input` produces a symbolic tensor (i.e. a placeholder). This symbolic tensor can be used with other
  TensorFlow ops, as such:

  ```python
  x = Input(shape=(32,))
  y = tf.square(x)
