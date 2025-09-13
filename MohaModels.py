import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor , ThreadPoolExecutor

from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
import sys
class Node:
    def __init__(self ,feature=None, threshold=None, left =None, right=None ,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def _is_leaf_node(self):
        return True if self.value else False
    def __repr__(self):
        if self._is_leaf_node():
            representation = f"Node(\n  value={self.value}\n    )"
        else:
            representation = f"Node(\n  feature={self.feature}\n  threshold={self.threshold}\n  left={self.left}\n  right={self.right}\n     )"

        return representation


class DecisionTreeRegressor:
    def __init__(self , max_depth=3 , min_sample_split=5 , n_features = 2 , random_state = 42):
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.random_state = random_state
        self.DecisionTree = None
        self.depth_counter = 0
        self.node_count = None

    def fit(self , X , y):
        self.n_features = self.n_features if (self.n_features <=X.shape[1]) and not self.n_features else X.shape[1]
        self.node_count = self._count_nodes()
        self.DecisionTree = self._grow_tree(X , y)
    def _count_nodes(self ):
        # node_count = 0
        # for i in range(1 , self.max_depth+1):
        #     node_count+=2**i
        return 2**(self.max_depth+1)-1

    def _grow_tree(self , X , y ,depth=0):

        self.depth_counter+=1


        n_samples , n_feats = X.shape

        if (depth>=self.max_depth or n_samples<self.min_sample_split):
            leaf_values = np.mean(y)
            return Node(value=leaf_values)
        else:



            # left = self._grow_tree(X[left_idx , :] , y[left_idx] , depth = depth+1)
            # right = self._grow_tree(X[right_idx , :] , y[right_idx] , depth = depth+1)

            # left_idx , right_idx = self._split(X[: , feature_to_split] ,splitting_threshold )
            # left_task = delayed(self._grow_tree)(X[left_idx, :], y[left_idx], depth=depth + 1)
            # right_task = delayed(self._grow_tree)(X[right_idx, :], y[right_idx], depth=depth + 1)

            # # Run both tasks in parallel
            # left, right = Parallel(n_jobs=6)([left_task, right_task])

            feat_idx = np.random.choice(n_feats , self.n_features , replace=False)
            feature_to_split , splitting_threshold = self._find_split(X,y , feat_idx)

            left_idx , right_idx = self._split(X[: , feature_to_split] ,splitting_threshold )
            
            with ThreadPoolExecutor(max_workers=2) as executor:


                # Create two tasks for left and right subtrees
                left_future = executor.submit(self._grow_tree, X[left_idx, :], y[left_idx], depth=depth+1)
                right_future = executor.submit(self._grow_tree, X[right_idx, :], y[right_idx], depth=depth+1)
                
                # Wait for both tasks to complete and get the result
                left = left_future.result()
                right = right_future.result()


            return Node(feature = feature_to_split , threshold = splitting_threshold , left = left , right = right)


    def _find_split(self , X, y , feat_idx):
        best_split , best_feature = None , None
        best_gain = -1


        for feature in feat_idx:
            feature_to_search_splits = X[: , feature]
            all_possible_thresholds = np.unique(feature_to_search_splits)
            for threshold in all_possible_thresholds:
                gain = self._calculate_gain(feature_to_search_splits , y , threshold)
                if gain>best_gain:
                    best_split = threshold
                    best_feature=feature
                    best_gain = gain
        
        return best_feature , best_split
        


    def _calculate_gain(self , feature_array , y , threshold):
        total_parent_samples = len(y)

        left_idx , right_idx = self._split(feature_array ,threshold )

        left_y , right_y = y[left_idx] , y[right_idx]


        left_value = np.mean(left_y)
        right_value = np.mean(right_y)
        parent_value = np.mean(y)

        left_variance = self._calculated_weighted_variance(left_y , left_value , total_parent_samples = total_parent_samples)
        right_variance = self._calculated_weighted_variance(right_y , right_value , total_parent_samples = total_parent_samples)
        parent_variance = self._calculated_weighted_variance(y , parent_value,total_parent_samples=None  )

        gain = parent_variance - (right_variance+left_variance)
        return gain
    
    def _split(self , feature_array , threshold):
        
        left_idx = np.argwhere(feature_array<=threshold).flatten()
        right_idx = np.argwhere(feature_array>threshold).flatten()

        return left_idx , right_idx


    def _calculated_weighted_variance(self , y_true , y_pred , total_parent_samples=None):
        length = len(y_true)
        weight = (length/total_parent_samples) if total_parent_samples else 1
        return np.mean(np.square(y_pred-y_true))*weight
    

    def predict(self , X):
        predictions = np.array([self._traverse_tree(x,self.DecisionTree ) for x in X])
        return predictions

    def _traverse_tree(self ,X , node):
        if node._is_leaf_node():
            return node.value
        else:
            if X[node.feature]<=node.threshold:
                return self._traverse_tree(X , node.left)
            else:

                return self._traverse_tree(X , node.right)
            
            