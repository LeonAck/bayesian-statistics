import numpy as np
from scipy.stats import bernoulli

from tree_simple import Tree
from bag import Bag

rng = np.random.default_rng(3)

class RandomForest(Bag):
    min_size = 1
    max_depth = 0

    def __init__(self, min_size, max_depth, n_trees, n_features):
        super().__init__(min_size, max_depth, n_trees)
        self.n_features = n_features

    def fit(self, X, Y):
        self.X, self.Y = X, Y
        n, p = self.X.shape
        for _ in range(self.n_trees):
            bootstrap = rng.integers(low=0, high=n, size=n)
            X, Y = self.X[bootstrap], self.Y[bootstrap]
            features = rng.choice(range(p), size=self.n_features, replace=False)
            tree = Tree(min_size = self.min_size, max_depth = self.max_depth)
            tree.fit(X[:, features], Y)
            self.trees.append(tree)

def test():
    X = np.array(
        [
            [2.771244718, 1.784783929],
            [1.728571309, 1.169761413],
            [3.678319846, 2.81281357],
            [3.961043357, 2.61995032],
            [2.999208922, 2.209014212],
            [7.497545867, 3.162953546],
            [9.00220326, 3.339047188],
            [7.444542326, 0.476683375],
            [10.12493903, 3.234550982],
            [6.642287351, 3.319983761],
        ]
    )
    Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    RandomForest.max_depth = 2
    RandomForest.min_size = 1
    rf = RandomForest(max_depth=2, min_size=1, n_trees=51, n_features=1)
    rf.fit(X, Y)

    tests = [[1, 10], [4, -5], [6, 1], [7, 2], [8, 5]]
    for x in tests:
        x = np.array(x).reshape(1,2)
        print(f"Predicted class of {x}: {rf.predict(x)}")

if __name__ == "__main__":
    test()
