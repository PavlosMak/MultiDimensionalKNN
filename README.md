# MultiDimensionalKNN
A k-nearest neighbours classifier that can operate on multidimensional data.

## How to use the classifier ##
You can find the following code in the example.py file.
To use the classifier first import it from the multiDKNN.py file: 
```
from multiDKNN import MultiDKNN
```
Then create an instance of the classifier and fit it to your data:
```
clf = MultiDKNN() 
clf.fit(x,y)
``` 
Here x must be a list-like object that contains list-like objects with the features of your data points
(e.g x = [[feature1,feature2,...],[feature1,feature2,...],...]). While y must be a list-like objects that contains your labels
(e.g y = [label1,label2,...]). After that predict the labels of unlabeled data using:
```
outcomes = clf.predict(test_x,k)
``` 
Here test_x must be a list-like object that either contains other list-like objects (like x) if you want to predict the labels of multiple data points, or just the coordinates of one data points (e.g test_x = [13.4,45.7,231.23,89.0]). K represents the number of nearest data points that will be considered in order to determine the label(s) of the new data-point(s).
If you want to calculate the accurasy of the classifier you can use:
```
accuracy =  clf.score(outcomes,y_test) 
``` 
where y_test muse be a list-like object that contains the labels of the data used to predict the outcomes.
In case you are not satisfied by the accuracy you can do:
```
best_k = clf.optimizek(test_x,test_y,up_limit) 
``` 
this will find the optimal value of K. Here test_x and test_y are as previously defined and up_limit is the highest value of K that will be tested.  
