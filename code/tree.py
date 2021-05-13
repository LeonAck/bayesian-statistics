import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs

import sklearn as sk

# import matplotlib.pyplot as plt


def p(Y, w, z):
    res = w[Y == z].sum()
    if res == 0:
        return 0
    return res / w.sum()


def missclassify(Y, w):
    return 1 - max(p(Y, w, -1), p(Y, w, 1))


def gini(Y, w):
    return 1 - p(Y, w, -1) ** 2 - p(Y, w, 1) ** 2


class Tree:
    def __init__(self, depth=0, min_size=1, max_depth=1):
        self.X, self.Y = None, None
        self.depth = depth
        self.left, self.right = None, None
        self.split_col = None
        self.split_value = None
        Tree.min_size = min_size
        Tree.max_depth = max_depth

    def size(self):
        return len(self.Y)

    def score(self, Y, w):
        # return missclassify(Y)
        return gini(Y, w)

    def score_of_split(self, i, j):
        s = self.X[:, j] <= self.X[i, j]
        l_score = self.score(self.Y[s], self.w[s]) * self.w[s].sum() / self.w.sum()
        r_score = self.score(self.Y[~s], self.w[~s]) * self.w[~s].sum() / self.w.sum()
        return l_score + r_score

    def find_optimal_split(self):
        n, p = self.X.shape
        best_score = self.score(self.Y, self.w)
        best_row = None
        best_col = None
        for j in range(p):
            for i in range(n):
                score = self.score_of_split(i, j)
                if score < best_score:
                    best_score, best_row, best_col = score, i, j
                    # print("best", i, j, score)
        self.split_row = best_row
        self.split_col = best_col
        self.split_value = self.X[best_row, best_col]

    def split(self):
        if self.size() <= self.min_size or self.depth >= self.max_depth:
            return
        self.find_optimal_split()
        if self.split_col == None:
            return
        s = self.X[:, self.split_col] <= self.split_value
        if s.all() or (~s).all():
            return
        self.left = Tree(depth=self.depth + 1)
        self.left.fit(self.X[s], self.Y[s])
        self.right = Tree(depth=self.depth + 1)
        self.right.fit(self.X[~s], self.Y[~s])

    def fit(self, X, Y, sample_weight=None):
        self.X, self.Y = X, Y
        if sample_weight is None:
            sample_weight = np.ones(len(Y))
        self.w = sample_weight / sample_weight.sum()
        self.split()

    def terminal(self):
        return self.left == None or self.right == None

    def majority_vote(self):
        if p(self.Y, self.w, -1) >= p(self.Y, self.w, 1):
            return -1
        return 1

    def _predict(self, x):
        if self.terminal():
            return self.majority_vote()
        if x[self.split_col] <= self.split_value:
            return self.left._predict(x)
        else:
            return self.right._predict(x)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])


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
    Y = 2 * Y - 1

    Tree.max_depth = 1
    Tree.min_size = 1
    tree = Tree()
    tree.fit(X, Y)
    print(tree.predict(X))


def test_2():
    X = np.arange(5).reshape(5, 1)
    Y = np.array([1, 0, 0, 1, 1])
    Y = 2 * Y - 1
    n, p = X.shape
    w = np.ones(n)
    w[0] = 100
    w /= w.sum()

    tree = Tree()
    tree.fit(X, Y, sample_weight=w)
    my_y = tree.predict(X)

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, Y, sample_weight=w)
    their_y = clf.predict(X)
    print((my_y == their_y).all())
    # print(clf.tree_.feature[0], clf.tree_.threshold[0], clf.tree_.impurity)
    # sk.tree.plot_tree(clf)
    # plt.show()


def test_3():
    np.random.seed(4)
    X, Y = make_blobs(n_samples=13, n_features=3, centers=2, cluster_std=20)
    Y = 2 * Y - 1
    n, p = X.shape
    w = np.random.uniform(size=n)
    w /= w.sum()

    tree = Tree()
    tree.fit(X, Y, sample_weight=w)
    my_y = tree.predict(X)

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, Y, sample_weight=w)
    their_y = clf.predict(X)
    print((my_y == their_y).all())


if __name__ == "__main__":
    # test()
    # test_2()
    test_3()
