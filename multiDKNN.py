class MultiDKNN():
    def __init__(self):
        self.x = [[]]
        self.y = []
        self.dimensions = 0
        
    def check_iterable(self,obj):
        '''Returns true if an objeck is iterable otherwise it return false. Meant for internal use'''
        try:
            obj[0]
            return True
        except:
            return False
    
    def check(self):
        '''Checks that the training data provide is of the correct type and shape. Meant for internal use'''
        if self.check_iterable(self.y) != True: #check that labels are in a list
            raise TypeError("The labels must be contained in a list-like object")
        if self.check_iterable(self.x) != True:
            raise TypeError("The features must be in a list-like object")
        if len(self.x) != len(self.y):
            raise Exception("The list-like object of features must be the same lenght as the list-like object of labels")
        
        for i in self.x:
            if self.check_iterable(i) != True:
                raise TypeError("The coordinates of each point must be in a list-like object")
        
        self.dimensions = len(self.x[0]) #find how many coordinates the points have
        
        for i in self.x: #check that all points have the same num of coordinates
            if len(i) != self.dimensions:
                raise Exception("All points need to have the same number of coordinates")    
        
    def fit(self,x,y): #x must be a list of lists(coordinates(features)), y must be a list(labels) 
        '''
        Fits the model. Takes two arguments: 
            x: Must be list-like object of list-like objects containing the features  
            y: Must be a list-like object containing the labels for each point
        ''' 
        self.x = x
        self.y = y
        self.check()
        
    def find_distance(self,i,j):
        '''Returns the distance of two n-dimensional vertices. Meant for internal use by the classifier'''
        s = 0
        for dimension in range(self.dimensions):
            s = s + (i[dimension] - j[dimension])**2
        distance = s**0.5
        return(distance)
    
    def argsort(self,l):
        '''Returns a list-like object containing the sorted indexes of a list-like object.Meant for internal use'''  
        sorted_list = sorted(l)
        sorted_indexes = []
        for el in sorted_list:
            sorted_indexes.append(l.index(el))
        return sorted_indexes
    
    def mode(self,l):
        '''Returns the mode(most common element) of a list-like object. Meant for internal use'''
        counter = {}
        most_common_element = 0
        most_common_element_vote = 0
        for i in l:
            if i not in counter:
                c = 0
                for e in l:
                    if e == i:
                        c += 1
            counter[i] = c
        for i in counter:
            if counter[i] > most_common_element_vote:
                most_common_element = i
                most_common_element_vote = counter[i]
        return most_common_element    
        
    def predict(self,p,k):
        '''
        Predicts the label of a new point or points. Takes 2 arguments:
            p: Unlabeled point/points must be a list-like object that either containes floating point values (for one point) or 
                list-like objects containing floating point values (for many points, each list represents a point)
            k: The number of nearest neighbors the classifier takes into consideration. Must be an integer
        '''
        if self.check_iterable(p) == True:
            if self.check_iterable(p[0]) != True:
                distances = []
                smallest_distances_indexes = []
                possible_labels = []
                for i in self.x:
                    distance = self.find_distance(p,i)
                    distances.append(distance)
                smallest_distances_indexes = self.argsort(distances)
                for i in smallest_distances_indexes[:k]:
                    possible_labels.append(self.y[i])   
                return self.mode(possible_labels)
            else:
                outcomes = []
                for point in p:
                    outcomes.append(self.predict(point,k))
                return outcomes
        else:
            raise TypeError("The coordinates of the unlabeled point/points must be in a list or other list-like object")
    
    def score(self,predicted_labels,true_labels):
        '''
        Returns a value for 0.0 to 1.0 which represents the accuracy of the classifier's results, the bigger the value the more accurate the classifier. 
        Takes 2 arguments:
            predicted_labels: A list-like object that contains the results of a classifier for a specific set of points. 
                              Use labeled points in the MultiDKNN().predict() to get those 
            true_labels: A list-like object that contains the valid labels of a specific set of points
        '''
            
        i = 0
        num_of_corrects = 0
        for label in predicted_labels:
            if label == true_labels[i]:
                num_of_corrects += 1
            i += 1
        return num_of_corrects/len(predicted_labels)
    
    def optimizek(self,p,true_labels,up_limit):
        '''
        Returns the optimal value of k for a specific dataset. Takes 3 arguments:
            p:unlabeled point/points must be a list-like object that either containes floating point values (for one point) or 
                list-like objects containing floating point values (for many points, each list represents a point)  
            true_labels: the label or labels for p. Must be a list-like object of equal lenght with p
            up_limit: The highest value of k to be examined. All values from 1 to up_limit(contained) will be checked. Must be an integer
        '''
        optimal_k = (0,0)
        for i in range(1,up_limit+1):
            outcomes = self.predict(p,i)
            current_accuracy = self.score(outcomes,true_labels)
            if current_accuracy > optimal_k[1]:
                optimal_k = (i,current_accuracy)
            if current_accuracy == 1.0:
                break
        return optimal_k[0]

#The following lines include the iris dataset by Fisher(1936).
class IrisDataset():
    '''
    Fisher's(1936) iris flower dataset formatted for imidiate use by the MultiDKNN classifier.
    FISHER, R. A. (1936), THE USE OF MULTIPLE MEASUREMENTS IN TAXONOMIC PROBLEMS. Annals of Eugenics, 7: 179-188. doi:10.1111/j.1469-1809.1936.tb02137.x
    '''
       
    feature_names = [
          ['sepal length (cm)',
            'sepal width (cm)',
            'petal length (cm)',
            'petal width (cm)'
          ]
        ]
    features = [
          [5.0999999999999996, 3.5, 1.3999999999999999, 0.20000000000000001],
          [4.9000000000000004, 3.0, 1.3999999999999999, 0.20000000000000001],
          [4.7000000000000002, 3.2000000000000002, 1.3, 0.20000000000000001],
          [4.5999999999999996, 3.1000000000000001, 1.5, 0.20000000000000001],
          [5.0, 3.6000000000000001, 1.3999999999999999, 0.20000000000000001],
          [5.4000000000000004, 3.8999999999999999, 1.7, 0.40000000000000002],
          [4.5999999999999996,
            3.3999999999999999,
            1.3999999999999999,
            0.29999999999999999
          ],
          [5.0, 3.3999999999999999, 1.5, 0.20000000000000001],
          [4.4000000000000004,
            2.8999999999999999,
            1.3999999999999999,
            0.20000000000000001
          ],
          [4.9000000000000004, 3.1000000000000001, 1.5, 0.10000000000000001],
          [5.4000000000000004, 3.7000000000000002, 1.5, 0.20000000000000001],
          [4.7999999999999998,
            3.3999999999999999,
            1.6000000000000001,
            0.20000000000000001
          ],
          [4.7999999999999998, 3.0, 1.3999999999999999, 0.10000000000000001],
          [4.2999999999999998, 3.0, 1.1000000000000001, 0.10000000000000001],
          [5.7999999999999998, 4.0, 1.2, 0.20000000000000001],
          [5.7000000000000002, 4.4000000000000004, 1.5, 0.40000000000000002],
          [5.4000000000000004, 3.8999999999999999, 1.3, 0.40000000000000002],
          [5.0999999999999996, 3.5, 1.3999999999999999, 0.29999999999999999],
          [5.7000000000000002, 3.7999999999999998, 1.7, 0.29999999999999999],
          [5.0999999999999996, 3.7999999999999998, 1.5, 0.29999999999999999],
          [5.4000000000000004, 3.3999999999999999, 1.7, 0.20000000000000001],
          [5.0999999999999996, 3.7000000000000002, 1.5, 0.40000000000000002],
          [4.5999999999999996, 3.6000000000000001, 1.0, 0.20000000000000001],
          [5.0999999999999996, 3.2999999999999998, 1.7, 0.5],
          [4.7999999999999998,
            3.3999999999999999,
            1.8999999999999999,
            0.20000000000000001
          ],
          [5.0, 3.0, 1.6000000000000001, 0.20000000000000001],
          [5.0, 3.3999999999999999, 1.6000000000000001, 0.40000000000000002],
          [5.2000000000000002, 3.5, 1.5, 0.20000000000000001],
          [5.2000000000000002,
            3.3999999999999999,
            1.3999999999999999,
            0.20000000000000001
          ],
          [4.7000000000000002,
            3.2000000000000002,
            1.6000000000000001,
            0.20000000000000001
          ],
          [4.7999999999999998,
            3.1000000000000001,
            1.6000000000000001,
            0.20000000000000001
          ],
          [5.4000000000000004, 3.3999999999999999, 1.5, 0.40000000000000002],
          [5.2000000000000002, 4.0999999999999996, 1.5, 0.10000000000000001],
          [5.5, 4.2000000000000002, 1.3999999999999999, 0.20000000000000001],
          [4.9000000000000004, 3.1000000000000001, 1.5, 0.10000000000000001],
          [5.0, 3.2000000000000002, 1.2, 0.20000000000000001],
          [5.5, 3.5, 1.3, 0.20000000000000001],
          [4.9000000000000004, 3.1000000000000001, 1.5, 0.10000000000000001],
          [4.4000000000000004, 3.0, 1.3, 0.20000000000000001],
          [5.0999999999999996, 3.3999999999999999, 1.5, 0.20000000000000001],
          [5.0, 3.5, 1.3, 0.29999999999999999],
          [4.5, 2.2999999999999998, 1.3, 0.29999999999999999],
          [4.4000000000000004, 3.2000000000000002, 1.3, 0.20000000000000001],
          [5.0, 3.5, 1.6000000000000001, 0.59999999999999998],
          [5.0999999999999996,
            3.7999999999999998,
            1.8999999999999999,
            0.40000000000000002
          ],
          [4.7999999999999998, 3.0, 1.3999999999999999, 0.29999999999999999],
          [5.0999999999999996,
            3.7999999999999998,
            1.6000000000000001,
            0.20000000000000001
          ],
          [4.5999999999999996,
            3.2000000000000002,
            1.3999999999999999,
            0.20000000000000001
          ],
          [5.2999999999999998, 3.7000000000000002, 1.5, 0.20000000000000001],
          [5.0, 3.2999999999999998, 1.3999999999999999, 0.20000000000000001],
          [7.0, 3.2000000000000002, 4.7000000000000002, 1.3999999999999999],
          [6.4000000000000004, 3.2000000000000002, 4.5, 1.5],
          [6.9000000000000004, 3.1000000000000001, 4.9000000000000004, 1.5],
          [5.5, 2.2999999999999998, 4.0, 1.3],
          [6.5, 2.7999999999999998, 4.5999999999999996, 1.5],
          [5.7000000000000002, 2.7999999999999998, 4.5, 1.3],
          [6.2999999999999998,
            3.2999999999999998,
            4.7000000000000002,
            1.6000000000000001
          ],
          [4.9000000000000004, 2.3999999999999999, 3.2999999999999998, 1.0],
          [6.5999999999999996, 2.8999999999999999, 4.5999999999999996, 1.3],
          [5.2000000000000002,
            2.7000000000000002,
            3.8999999999999999,
            1.3999999999999999
          ],
          [5.0, 2.0, 3.5, 1.0],
          [5.9000000000000004, 3.0, 4.2000000000000002, 1.5],
          [6.0, 2.2000000000000002, 4.0, 1.0],
          [6.0999999999999996,
            2.8999999999999999,
            4.7000000000000002,
            1.3999999999999999
          ],
          [5.5999999999999996, 2.8999999999999999, 3.6000000000000001, 1.3],
          [6.7000000000000002,
            3.1000000000000001,
            4.4000000000000004,
            1.3999999999999999
          ],
          [5.5999999999999996, 3.0, 4.5, 1.5],
          [5.7999999999999998, 2.7000000000000002, 4.0999999999999996, 1.0],
          [6.2000000000000002, 2.2000000000000002, 4.5, 1.5],
          [5.5999999999999996, 2.5, 3.8999999999999999, 1.1000000000000001],
          [5.9000000000000004, 3.2000000000000002, 4.7999999999999998, 1.8],
          [6.0999999999999996, 2.7999999999999998, 4.0, 1.3],
          [6.2999999999999998, 2.5, 4.9000000000000004, 1.5],
          [6.0999999999999996, 2.7999999999999998, 4.7000000000000002, 1.2],
          [6.4000000000000004, 2.8999999999999999, 4.2999999999999998, 1.3],
          [6.5999999999999996, 3.0, 4.4000000000000004, 1.3999999999999999],
          [6.7999999999999998,
            2.7999999999999998,
            4.7999999999999998,
            1.3999999999999999
          ],
          [6.7000000000000002, 3.0, 5.0, 1.7],
          [6.0, 2.8999999999999999, 4.5, 1.5],
          [5.7000000000000002, 2.6000000000000001, 3.5, 1.0],
          [5.5, 2.3999999999999999, 3.7999999999999998, 1.1000000000000001],
          [5.5, 2.3999999999999999, 3.7000000000000002, 1.0],
          [5.7999999999999998, 2.7000000000000002, 3.8999999999999999, 1.2],
          [6.0, 2.7000000000000002, 5.0999999999999996, 1.6000000000000001],
          [5.4000000000000004, 3.0, 4.5, 1.5],
          [6.0, 3.3999999999999999, 4.5, 1.6000000000000001],
          [6.7000000000000002, 3.1000000000000001, 4.7000000000000002, 1.5],
          [6.2999999999999998, 2.2999999999999998, 4.4000000000000004, 1.3],
          [5.5999999999999996, 3.0, 4.0999999999999996, 1.3],
          [5.5, 2.5, 4.0, 1.3],
          [5.5, 2.6000000000000001, 4.4000000000000004, 1.2],
          [6.0999999999999996, 3.0, 4.5999999999999996, 1.3999999999999999],
          [5.7999999999999998, 2.6000000000000001, 4.0, 1.2],
          [5.0, 2.2999999999999998, 3.2999999999999998, 1.0],
          [5.5999999999999996, 2.7000000000000002, 4.2000000000000002, 1.3],
          [5.7000000000000002, 3.0, 4.2000000000000002, 1.2],
          [5.7000000000000002, 2.8999999999999999, 4.2000000000000002, 1.3],
          [6.2000000000000002, 2.8999999999999999, 4.2999999999999998, 1.3],
          [5.0999999999999996, 2.5, 3.0, 1.1000000000000001],
          [5.7000000000000002, 2.7999999999999998, 4.0999999999999996, 1.3],
          [6.2999999999999998, 3.2999999999999998, 6.0, 2.5],
          [5.7999999999999998,
            2.7000000000000002,
            5.0999999999999996,
            1.8999999999999999
          ],
          [7.0999999999999996, 3.0, 5.9000000000000004, 2.1000000000000001],
          [6.2999999999999998, 2.8999999999999999, 5.5999999999999996, 1.8],
          [6.5, 3.0, 5.7999999999999998, 2.2000000000000002],
          [7.5999999999999996, 3.0, 6.5999999999999996, 2.1000000000000001],
          [4.9000000000000004, 2.5, 4.5, 1.7],
          [7.2999999999999998, 2.8999999999999999, 6.2999999999999998, 1.8],
          [6.7000000000000002, 2.5, 5.7999999999999998, 1.8],
          [7.2000000000000002, 3.6000000000000001, 6.0999999999999996, 2.5],
          [6.5, 3.2000000000000002, 5.0999999999999996, 2.0],
          [6.4000000000000004,
            2.7000000000000002,
            5.2999999999999998,
            1.8999999999999999
          ],
          [6.7999999999999998, 3.0, 5.5, 2.1000000000000001],
          [5.7000000000000002, 2.5, 5.0, 2.0],
          [5.7999999999999998,
            2.7999999999999998,
            5.0999999999999996,
            2.3999999999999999
          ],
          [6.4000000000000004,
            3.2000000000000002,
            5.2999999999999998,
            2.2999999999999998
          ],
          [6.5, 3.0, 5.5, 1.8],
          [7.7000000000000002,
            3.7999999999999998,
            6.7000000000000002,
            2.2000000000000002
          ],
          [7.7000000000000002,
            2.6000000000000001,
            6.9000000000000004,
            2.2999999999999998
          ],
          [6.0, 2.2000000000000002, 5.0, 1.5],
          [6.9000000000000004,
            3.2000000000000002,
            5.7000000000000002,
            2.2999999999999998
          ],
          [5.5999999999999996, 2.7999999999999998, 4.9000000000000004, 2.0],
          [7.7000000000000002, 2.7999999999999998, 6.7000000000000002, 2.0],
          [6.2999999999999998, 2.7000000000000002, 4.9000000000000004, 1.8],
          [6.7000000000000002,
            3.2999999999999998,
            5.7000000000000002,
            2.1000000000000001
          ],
          [7.2000000000000002, 3.2000000000000002, 6.0, 1.8],
          [6.2000000000000002, 2.7999999999999998, 4.7999999999999998, 1.8],
          [6.0999999999999996, 3.0, 4.9000000000000004, 1.8],
          [6.4000000000000004,
            2.7999999999999998,
            5.5999999999999996,
            2.1000000000000001
          ],
          [7.2000000000000002, 3.0, 5.7999999999999998, 1.6000000000000001],
          [7.4000000000000004,
            2.7999999999999998,
            6.0999999999999996,
            1.8999999999999999
          ],
          [7.9000000000000004, 3.7999999999999998, 6.4000000000000004, 2.0],
          [6.4000000000000004,
            2.7999999999999998,
            5.5999999999999996,
            2.2000000000000002
          ],
          [6.2999999999999998, 2.7999999999999998, 5.0999999999999996, 1.5],
          [6.0999999999999996,
            2.6000000000000001,
            5.5999999999999996,
            1.3999999999999999
          ],
          [7.7000000000000002, 3.0, 6.0999999999999996, 2.2999999999999998],
          [6.2999999999999998,
            3.3999999999999999,
            5.5999999999999996,
            2.3999999999999999
          ],
          [6.4000000000000004, 3.1000000000000001, 5.5, 1.8],
          [6.0, 3.0, 4.7999999999999998, 1.8],
          [6.9000000000000004,
            3.1000000000000001,
            5.4000000000000004,
            2.1000000000000001
          ],
          [6.7000000000000002,
            3.1000000000000001,
            5.5999999999999996,
            2.3999999999999999
          ],
          [6.9000000000000004,
            3.1000000000000001,
            5.0999999999999996,
            2.2999999999999998
          ],
          [5.7999999999999998,
            2.7000000000000002,
            5.0999999999999996,
            1.8999999999999999
          ],
          [6.7999999999999998,
            3.2000000000000002,
            5.9000000000000004,
            2.2999999999999998
          ],
          [6.7000000000000002, 3.2999999999999998, 5.7000000000000002, 2.5],
          [6.7000000000000002, 3.0, 5.2000000000000002, 2.2999999999999998],
          [6.2999999999999998, 2.5, 5.0, 1.8999999999999999],
          [6.5, 3.0, 5.2000000000000002, 2.0],
          [6.2000000000000002,
            3.3999999999999999,
            5.4000000000000004,
            2.2999999999999998
          ],
          [5.9000000000000004, 3.0, 5.0999999999999996, 1.8]
        ]
    labels = ['setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'setosa',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'versicolor',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica',
          'virginica'
        ]