from multiDKNN import MultiDKNN, IrisDataset #Imports the classifier and the iris dataset

data = IrisDataset() #Creates an instance of the IrisDataset
feature_names = data.feature_names #This will store the names of the features.
x_train = data.features[:140] #Holds the features of the dataset that will be used for training
y_train = data.labels[:140]  #Holds the labels of the features of the dataset that will be used for training
x_test = data.features[140:] #Holds the features that will be used for testing
y_test = data.labels[140:] #Holds the labels of the features that will be used for testing

clf = MultiDKNN() #Creates an instance of the MultiDKNN() classifier
clf.fit(x_train,y_train) #Fits the classifier
outcomes = clf.predict(x_test,5) #Predicts the labels of the x_test data
accuracy = clf.score(outcomes,y_test) #Calculates the accuracy of the classifier
best_k = clf.optimizek(x_test,y_test,10) #Finds the optimal value of k for the specific set of features and labels 