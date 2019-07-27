# Python implementation of the paper "A General Reflex Fuzzy Min-Max Neural Network."

One Paragraph of project description goes here

### Prerequisites
Python3: numpy, pandas, sklearn

### Installing
```
 pip install numpy pandas sklearn
```

## Running the Network

### Import Iris data, split, and scale

The network is tested with the Iris data set. The training samples and test samples, X, need to be a numpy array with 
the shape of (sample,features). The labels of the training samples and test samples, Y, need the class labels 
to be assigned to natural numbers (1, 2, 3, 4, .....) and also be in the form of a numpy array. The values of X need to be scalled to the interval of (0,1).

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

The network is tested with the Iris data set. The training samples and test samples, X, need to be a numpy array with 
the shape of (sample,features). The labels of the training samples and test samples, Y, need the class labels 
to be assigned to natural numbers (1, 2, 3, 4, .....) and also be in the form of a numpy array.

```
    # --- Declare network --- #
    nn = ReflexFuzzyNeuroNetwork(gamma=2, theta=.1)
```

### Train and Test Network

The network is tested with the Iris data set. The training samples and test samples, X, need to be a numpy array with 
the shape of (sample,features). The labels of the training samples and test samples, Y, need the class labels 
to be assigned to natural numbers (1, 2, 3, 4, .....) and also be in the form of a numpy array.

```
    # --- Train network --- #
    nn.train(X_train, y_train)

    # --- Test Network --- #
    nn.test(X_test,y_test)
```


## Authors

* **Enrique Nueve** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* Nandedkar A.V., Biswas P.K. for creating the GRMMFN.
