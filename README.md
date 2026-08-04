<H3>NAME: AKASH A </H3>
<H3>REG.NO.: 212225240006 </H3>

<H3>Experiment No. 2 </H3>

## Implementation of Perceptron for Binary Classification

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron. <BR>
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’
f(x)=w.x+b
 <BR>
A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

 


<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">


Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>


# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
 ~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("student_pass.csv")
print(df.head())

# extract the label column
y = df.iloc[:,2].values

# extract features
x = df.iloc[:,0:2].values

plt.scatter(x[y==0,0], x[y==0,1], color='red', marker='o', label='Fail')
plt.scatter(x[y==1,0], x[y==1,1], color='green', marker='x', label='Pass')

plt.xlabel("Hours Studied")
plt.ylabel("Attendance")
plt.legend(loc='upper left')
plt.show()

y = np.where(y == 'Iris-Setosa',1,-1)
y = np.where(y == 1,1,-1)

# Convert labels
y = np.where(y == 1, 1, -1)

# Standardize features
x[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

# Split the data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0
)

# Train the model
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)

# Accuracy
print("Accuracy:", accuracy_score(y_test, classifier.predict(x_test)) * 100)

# Plot the number of errors during each iteration
plt.plot(
    range(1, len(classifier.misclassified_samples) + 1),
    classifier.misclassified_samples,
    marker='o'
)
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.show()

~~~


# OUTPUT:

<img width="557" height="111" alt="image" src="https://github.com/user-attachments/assets/88438472-c8b4-4efe-8e16-3fbf02ee37da" />

<img width="550" height="418" alt="image" src="https://github.com/user-attachments/assets/4e28e06a-04ca-4644-be6a-e7a02fd397d4" />

<img width="551" height="430" alt="image" src="https://github.com/user-attachments/assets/a1cdca9c-cc5f-4bd6-9580-b16557e0e00d" />



# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.

 
