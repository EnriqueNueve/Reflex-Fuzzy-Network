# Python implementation of the paper "A General Reflex Fuzzy Min-Max Neural Network."

One Paragraph of project description goes here

### Prerequisites
Python3: numpy, pandas, sklearn

### Installing
```
 pip install numpy pandas sklearn
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
    # --- Import Iris data and split --- #
    data = pd.read_csv('iris_data_norm.csv')
    data = data.iloc[:,1:]
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    scaler_min_max = MinMaxScaler(feature_range=(0.01, .99))
    X = scaler_min_max.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    y_train, y_test = y_train.values, y_test.values
    X_train, X_test = X_train.T, X_test.T

    # --- Declare network --- #
    nn = ReflexFuzzyNeuroNetwork(gamma=2, theta=.1)

    # --- Run network --- #
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
