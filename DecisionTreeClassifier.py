import numpy as np

class Node:
    """ Define a node object """
    def __init__(self, num_samples_per_class, predicted_class):
          self.num_samples_per_class = num_samples_per_class
          self.predicted_class = predicted_class
          self.feature_index = 0
          self.threshold = 0
          self.left = None
          self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth = 1e10, min_info_gain = None):
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain

    def _best_split(self, X, y):
        # Return None, None if only 1 observations in 'y'
        m = y.shape[0]
        if m <= 1:
            return None, None
        
        # Get count of all classes in current node, e.g. # [914, 586]
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)] 

        # Calculate gini of current node [914, 586] -> 0.522
        if self.impurity == "Gini":
            best_score = 1.0 - sum((n/m) ** 2 for n in num_parent)
        elif self.impurity == "Entropy":
            best_score = -sum((n/m) * np.log2(n/m) for n in num_parent)

        best_idx, best_thr = None, None

        # Search for the feature / threshold with best gini impurity (w. av of child nodes)
        for idx in range(self.n_features_):
            sorted_feature_idx = np.argsort(X[:,idx])
            thresholds = X[:,idx][sorted_feature_idx]
            classes = y[sorted_feature_idx]

            # Define 'left' and 'right' for a split, e.g. [0,0] [914, 586]
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()

            # It is possible to check every threshold, counting the class observations on each side of the 
            # threshold. This is quadratic. A linear approach is to start with all counts / class on the 'right'
            # and as we increase the threshold, we move counts / class over to the 'left'

            for i in range(1,m):
                c = classes[i - 1]
                num_left[c[0]] += 1 # becomes [0,1]
                num_right[c[0]] -= 1 # becomes [914, 585]

                # Calculate gini_left & gini_right & gini impurity of a split
                if self.impurity == "Gini":
                    score_left = 1.0 - sum((num_left[x]/i)**2 for x in range(self.n_classes_))
                    score_right = 1.0 - sum((num_right[x]/(m - i))**2 for x in range(self.n_classes_))

                elif self.impurity == "Entropy":
                    score_left = -sum(num_left[x]/i * np.log2(num_left[x]/i) for x in range(self.n_classes_))
                    score_right = -sum(num_right[x]/(m-i) * np.log2(num_right[x]/(m-i)) for x in range(self.n_classes_))

                score = (1 * score_left + (m - i) * score_right) / m
                
                # We can't have two values on different side of a split
                if thresholds[i] == thresholds[i - 1]:
                    continue

                # Lower gini scores are better
                if score < best_score:
                    best_score = score
                    best_score_left = score_left
                    best_score_right = score_right
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        # Calculate impurity gain
        if self.min_info_gain:
            if self.impurity == "Gini":
                information_gain = (1.0 - sum((n/m) ** 2 for n in num_parent)) - (best_score_left + best_score_right)
            elif self.impurity == "Entropy":
                information_gain = -sum((n/m) * np.log2(n/m) for n in num_parent) - (best_score_left + best_score_right)

            if information_gain < self.min_info_gain:
                return None, None

        return best_idx, best_thr


    def fit(self, X, y, impurity = "Gini"):
        """ Build decision tree classifier """
        self.n_classes_ = len(np.unique(y)) # Classes assumed to go from 0 to n-1
        self.n_features_ = X.shape[1] # Observations in columns
        self.impurity = impurity
        self.tree_ = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth = 0):
        """ Build a decision tree by recursively finding the best split """

        # Populate data for (first) node
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)

        # Pass the number of samples per class and predicted class into a node attributes
        node = Node(num_samples_per_class, predicted_class)

        # Apply recursion
        if depth < self.max_depth:
            idx, thr = self._best_split(X,y)

            # Create left and right nodes
            if idx is not None:

                # Return 'True' for values below the threshold
                indices_left = X[:, idx] < thr
                x_left, y_left = X[indices_left], y[indices_left]
                x_right, y_right = X[~indices_left], y[~indices_left]

                node.feature_index = idx
                node.threshold = thr

                # Recursion; pass new X,y values from left and right to create new nodes.
                node.left = self._grow_tree(x_left, y_left, depth + 1)
                node.right = self._grow_tree(x_right, y_right, depth + 1)

        return node


    def predict(self, X):
        " Return predicts on an matrix of inputs "
        return [self._predict(inputs) for inputs in X]


    def _predict(self, inputs):
        """ Predict class for a single input """
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


    def accuracy(self, preds, y):
        """ Returns accuracy analysis on prediction results vs. actual observations """
        num_pred = len(preds)

        TP = sum([1 for i in range(num_pred) if (preds[i] == 1 and y[i] == 1)  ])
        TN = sum([1 for i in range(num_pred) if (preds[i] == 0 and y[i] == 0)  ])
        FP = sum([1 for i in range(num_pred) if (preds[i] == 1 and y[i] == 0)  ])
        FN = sum([1 for i in range(num_pred) if (preds[i] == 0 and y[i] == 1)  ])

        assert(TP + TN + FP + FN == num_pred)

        return (TP + TN) / num_pred