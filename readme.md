## Decision Tree and Random Forrest Classifier using Numpy
Construct either a decision tree or a random forest for classification tasks. Both models have customisable pruning and impurity parameters and use numpy for its computations.
  
### Purpose
The purpose for this project was to better understand decision trees and random forests by building these models from scratch. This models can be applied to a broad array of classification problems.

### Features
* Customisability - DecisionTree has maximum depth and minimum information gain parameters to control tree growth. The impurity measure can also be specified. RandomForest also contains these parameters, but the number of trees, sample ratio (% samples used in each tree) and feature ratio (% features used in each tree) can also be specified.
* Predictions / accuracy - Both DecisionTree and RandomForest have predict and accuracy functions that can generate predictions and the accuracy of those predictions respectively.

### Usage - DecisionTreeClassifer  
  
(Class) **DecisionTreeClassifier(max_depth = 1e10, min_info_gain = None)** will instantiate a DecisionTreeClassifier object. max_depth and min_info_gain can be used to control tree growth. 

(Function) **.fit(X, y, impurity)** will train a decision tree. The parameters are as follows:  
* X: numpy array, rows = observations, columns = features.
* y: numpy array, rows =  class observations (0,1,2..), columns = 1.
* impurity: EIther "Gini" or "Entropy".

(Function) **.predict(X)** will return class predictions for a numpy array of observations X.
  
(Function) **.accuracy(preds, y)** will take the predicted observations and their true class and return the accuracy.

Example    
model = DecisionTreeClassifier()  
model.fit(X,y, "Gini")  
preds = model.predict(a)  
accuracy = model.accuracy(preds, b)  
  
### Usage - RandomForestClassifer
  
(Class) **RandomForestClassifier(max_depth = 1e10, min_info_gain = None)** will instantiate a RandomForestClassifier object. max_depth and min_info_gain can be used to control each tree growth in the forest.

(Function) **.fit(X, y, num_trees = 1000, feature_ratio = 0.1, sample_ratio = 0.1, impurity = "Gini")** will train a decision tree. The parameters are as follows:  
* X: numpy array, rows = observations, columns = features.
* y: numpy array, rows =  class observations (0,1,2..), columns = 1.
* impurity: EIther "Gini" or "Entropy".

(Function) **.predict(X)** will return class predictions for a numpy array of observations X. For each input it will make a prediction from each decision tree, before returning the most common prediction for each input.
  
(Function) **.accuracy(preds, y)** will take the predicted observations and their true class and return the accuracy.
  
Example    
model = RandomForestClassifier()  
model.fit(X, y, num_trees = 1000, "Gini")  
preds = model.predict(a)  
accuracy = model.accuracy(preds, b)  

### credits
Inspiration for parts of the code was used from https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775 . Although this code explicitly focuses on decision trees and Gini as an impurity measure. 


