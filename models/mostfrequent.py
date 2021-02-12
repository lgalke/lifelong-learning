import numpy as np

class MostFrequentClass():
    def __init__(self):
        self.most_frequent_class = None
        self.n_classes = None

    def fit(self, __X, y):
        unique_elements, counts_elements = np.unique(np.asarray(y), return_counts=True)
        self.most_frequent_class = unique_elements[np.argmax(counts_elements)]
        self.n_classes = unique_elements.size
        print("Most frequent class:", self.most_frequent_class)

    def predict(self, X):
        if None in [self.most_frequent_class, self.n_classes]:
            raise ValueError
        y = np.zeros((1, self.n_classes))
        y[:, self.most_frequent_class] = 1
        return np.repeat(y, X.shape[0], axis=0)

    def __call__(self, graph, feats):
        return self.predict(feats)

    def train(self):
        pass

    def eval(self):
        pass

    def reset_parameters(self):
        pass

    def reset_final_parameters(self):
        pass

    def final_parameters(self):
        yield
