# MultiDimensionalKNN
A k-nearest neighbours classifier that can operate on multidimensional data. It is created from scratch so no external modules are needed.

## How to use the classifier ##
You can find the following code in the [example.py](./example.py) file.
To use the classifier first import it from the [multiDKNN.py](./multiDKNN.py) file: 
```
from multiDKNN import MultiDKNN
```
Then create an instance of the classifier and fit it to your data:
```
clf = MultiDKNN() 
clf.fit(x_train,y_train)
``` 
Here x_train must be a list-like object that contains list-like objects with the features of your data points
(e.g x_train = [[feature1,feature2,...],[feature1,feature2,...],...]). While y_train must be a list-like objects that contains your labels
(e.g y_train = [label1,label2,...]). After that predict the labels of unlabeled data using:
```
outcomes = clf.predict(x_test,k)
``` 
Here x_test must be a list-like object that either contains other list-like objects (like x_train) if you want to predict the labels of multiple data points, or just the coordinates of one data points (e.g test_x = [13.4,45.7,231.23,89.0]). K represents the number of nearest data points that will be considered in order to determine the label(s) of the new data-point(s).
If you want to calculate the accuracy of the classifier you can use:
```
accuracy =  clf.score(outcomes,y_test) 
``` 
where y_test must be a list-like object that contains the labels of the data used to predict the outcomes.
In case you are not satisfied by the accuracy you can do:
```
best_k = clf.optimizek(x_test,y_test,up_limit) 
``` 
this will find the optimal value of K. Here test_x and test_y are as previously defined and up_limit is the highest value of K that will be tested. This method actually brute-forces all values of K up to up_limit so it might take a while.

## How to use the included dataset ##
The [multiDKNN.py](./multiDKNN.py) file includes Fisher's(1936) iris dataset, formatted for imidiate use from the MultiDimensionalKNN classifier. The following code can be found in the [example.py](./example.py) file. 
First of all import the dataset and create an instance of it:
```
from multiDKNN import IrisDataset()
data = IrisDataset()
``` 
Then you can seperate training and testing data with the following: 
```
x_train = data.features[:140] 
y_train = data.labels[:140]  
x_test = data.features[140:] 
y_test = data.labels[140:]
```
This will results in the creation of four lists:
  - x_train: A list of lists containig the coordinates of the first 140 datapoints.
  - y_train: A list containing the labels of the first 140 datapoints.
  - x_test: A list of lists containing the coordinates of the last 10 datapoints.
  - y_test: A list containing the labels of the last 10 datapoints.

Of course you can change 140 to any number you like or use a more sophisticated method to get shuffled testing data.
In case you want to see what each number in the features represents just run:
```
data.feature_names
``` 
## References ##
FISHER, R. A. (1936), THE USE OF MULTIPLE MEASUREMENTS IN TAXONOMIC PROBLEMS. Annals of Eugenics, 7: 179-188. doi:10.1111/j.1469-1809.1936.tb02137.x
























