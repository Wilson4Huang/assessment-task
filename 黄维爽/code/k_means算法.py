import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, n):
        # 簇数
        self.n_clusters = n
        self.centers = None

    def fit(self, X):
        # 参数检查
        assert isinstance(X, np.ndarray) and len(X.shape) == 2
        # 初始随机选取聚类中心(质心)
        indices = np.arange(X.shape[0])
        selected = np.random.choice(indices, self.n_clusters, replace=False)
        centers = X[selected]
        last_assignments = None
        num_iterations = 0
        while True:  #不断地聚类，不断地寻找聚类中心，直到中心不改变
            num_iterations += 1
            assignments = self.assign(X, centers)
            centers = self._update_centers(X, self.n_clusters, assignments)
           
            if last_assignments is not None and (last_assignments == assignments).all():  #分配了的同时，聚类结果不改变时退出迭代
                break
            last_assignments = assignments   #更新分配结果
        self.centers = centers   #更新簇中心

    def _update_centers(self, X, k, assignments):
        """更新簇中心"""
        """
            assignments为每个样本被分配好的簇下标
            k为簇数
        """
        mapping = {i: [] for i in range(k)}  #每个簇对样本的映射（对应关系）
        for i, p in zip(assignments, X):
            mapping[i].append(p)

        N, D = X.shape
        center_points = np.zeros((k, D))
        for i, points in mapping.items():
            # 计算属于这个簇的样本的均值作为簇中心（利用坐标的均值）
            center_points[i] = np.mean(np.array(points), axis=0)
        return center_points

    def assign(self, X, centers):
        """给定样本矩阵和簇中心，计算每个样本被分配的簇下标"""
        assignments = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # 最小距离
            dists = np.linalg.norm(X[i] - centers, axis=1)  #线性回归范数（平方和开根号：欧拉距离）  其中dists中有n（簇数）个
            assignments[i] = np.argmin(dists)      #每个样本都会分配到离自己最近的簇下标
        return assignments

    def predict(self, X):
        assignments = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # 最小距离
            dists = np.linalg.norm(X[i] - self.centers, axis=1)   #欧拉距离
            assignments[i] = np.argmin(dists)
        return assignments
