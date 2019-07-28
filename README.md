# Python implementation of the paper "A General Reflex Fuzzy Min-Max Neural Network."
This repository is a python3 implementation of the paper "A General Reflex Fuzzy Min-Max Neural Network" 
by Nandedkar A.V., Biswas P.K. The model, General Reflex Fuzzy Min-Max Neural Network, is a classifying fuzzy
model. Due to its unique design of having two extra hyperbox variants, an overlapping compensation neuron, and a containment compensation neuron, this fuzzy network has been shown to outpeform traditional fuzzy networks with only classifying neuron hyperboxes. 


### Prerequisites
Python3: numpy, pandas, sklearn

### Installing
```
 pip install numpy pandas sklearn
```

## Running the Network

### Import Iris data, split, and scale

The network is tested with the Iris data set. The training samples and test samples, X, need to be a numpy array with the shape of (sample, features). The labels of the training samples and test samples, Y, need the class labels to be assigned to natural numbers (1, 2, 3, 4, .....) and also be in the form of a numpy array. The values of X need to be scaled to the interval of (0,1).

```
    # --- Import Iris data, split, and scale --- #
    data = pd.read_csv('iris_data_norm.csv')
    data = data.iloc[:,1:]
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
    X = scaler_min_max.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    y_train, y_test = y_train.values, y_test.values
    X_train, X_test = X_train.T, X_test.T  
```

### Declare network
To use the network, it needs to be declared by calling the class "ReflexFuzzyNeuroNetwork" from the file GRFMMN.py. Two tuning parameters are passed during the declaration, gamma, and theta. Gamma serves as a tuning factor for the sensitivity of the membership function and theta serves a tuning factor for the expansion criteria. Ranges for these values most often fall within [2,4] for gamma and [.1,1] for theta.

```
    # --- Declare network --- #
    nn = ReflexFuzzyNeuroNetwork(gamma=2, theta=.1)
```

### Train and Test Network
The X inputs and Y labels need to be separated into two separate date sets, a training and test set. The X-values
are the first passed parameter for "train" and "test" functions and, the y-values are the second parameter passed for the 
"train" and "test" functions. 

```
    # --- Train network --- #
    nn.train(X_train, y_train)

    # --- Test Network --- #
    nn.test(X_test,y_test)
```

The test function will print out the accuracy of the model for predicting the test set.
```
Accuracy: 98.0% 
```


## Authors

* **Enrique Nueve** 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* Nandedkar A.V., Biswas P.K. for creating the GRMMFN.
