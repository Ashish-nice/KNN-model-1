#Making necessary imports
import numpy as np

class NearestNeighbour:
    def __init__(self):
        self.k = 3

    #Training the model is just adding data to the dataset O(1)
    def train(self,X,y):
        self.Xtr = X
        self.ytr = y

    #Testing the model is finding the distance between the datapoint and the existing data trained O(N)
    def test(self,X):
        num_test = X.shape[0]
        y_pred = np.zeros(num_test,dtype=self.ytr.dtype)

        #looping over all the test data
        for i in range(num_test):
            #Find the k nearest training images to the ith test data
            #using L2 distance
            distances= np.linalg.norm(self.Xtr-X[i,:],axis=1)
            min_indices = np.argpartition(distances,self.k)[:self.k]
            unique_val,counts = np.unique(min_indices,return_counts=True)
            max_count = np.max(counts)
            modes = unique_val[counts==max_count]
            if(modes.size>1):
                mode = modes[np.argmin(distances[modes])]
            y_pred[i] = self.ytr[mode]#Predict the label of the specefic training data
        return y_pred


