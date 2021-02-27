import numpy as np
from collections import deque
from numpy.linalg import eig, inv 


def euclidean_distance(x, y):
    sq_x = np.sum(x ** 2, axis=-1)
    sq_y = np.sum(y ** 2, axis=-1)
    z = -2 * np.dot(x, np.transpose(y))
    z = z + sq_x.reshape(-1, 1) + sq_y.reshape(1, -1)
    z = np.sqrt(z)
    return z


def cosine_distance(x, y):
    dots = np.dot(x, np.transpose(y))
    sq_x = np.sqrt(np.sum(x ** 2, axis=-1))
    sq_y = np.sqrt(np.sum(y ** 2, axis=-1))
    return 1 - dots / sq_x.reshape(-1, 1) / sq_y.reshape(1, -1)


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Neighbor metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        d = self._metric_func(self._X, X)
        index_array = np.argpartition(d, kth=self.n_neighbors, axis=0)[:self.n_neighbors, :]
        index_array = np.take_along_axis(index_array, np.argsort(np.take_along_axis(d, index_array, axis=0),
                                                                 axis=0), axis=0)
        if return_distance:
            return (np.transpose(np.take_along_axis(d, index_array, axis=0)),
                    np.transpose(index_array))
        else:
            return np.transpose(index_array)

        
class LaplacianEigenmaps():
    def __init__(self, t, method, m, eps=None, n=None, kernel_metric="euclidean", neighbor_metric="euclidean"):
        if kernel_metric == "euclidean":
            self.kernel_metric = kernel_metric
        else:
            raise ValueError("Kernel metric is not supported", metric)
        if method == "a":
            self.method = "a"
            assert eps is not None
            self.eps = eps
        elif method == "b":
            self.method = "b"
            assert n is not None
            self.n = n
        else:
            raise ValueError("Method is not supported, use \"a\" or \"b\"", method)
        self.t = t
        self.m = m
        self.neighbor_metric = neighbor_metric
        
    def heat_kernel(self, x1, x2):
        if self.t is np.inf:
            return 1
        if self.kernel_metric == "euclidean":
            return np.exp(np.sum((x1 - x2) ** 2) / self.t)

    def construct_graph(self, X):
        self.W = np.zeros(shape=(X.shape[0], X.shape[0]))
        if self.method == "a":
            for i in range(X.shape[0]):
                for j in range(i, X.shape[0]):
                    if np.sum((X[i, :] - X[j, :]) ** 2) < self.eps:
                        self.W[i, j] = heat_kernel(X[i, :], X[j, :])
                        self.W[j, i] = heat_kernel(X[i, :], X[j, :])
        elif self.method == "b":
            knn = NearestNeighborsFinder(self.n, metric=self.neighbor_metric)
            knn.fit(X)
            kneighbors = knn.kneighbors(X)
            for i in range(kneighbors.shape[0]):
                for j in kneighbors[i, :]:
                    self.W[i, j] = self.heat_kernel(X[i, :], X[j, :])
                    self.W[j, i] = self.heat_kernel(X[i, :], X[j, :])
        return self.W
    
    
    def _adjacency_matrix_to_adjacency_list(self, W=None):
        self.W_list = [[] for i in range(self.W.shape[0])]
        if W is None:
            W = self.W
        for i in range(self.W.shape[0]):
            for ind, j in enumerate(W[i, :]):
                if j != 0 and ind != i:
                    self.W_list[i].append(ind)
        return self.W_list
    
    
    def connected_components(self):
        seen = set()
        graph = self._adjacency_matrix_to_adjacency_list()
        for root in range(len(graph)):
            if root not in seen:
                seen.add(root)
                component = []
                queue = deque([root])
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in graph[node]:
                        if neighbor not in seen:
                            seen.add(neighbor)
                            queue.append(neighbor)
                yield component
    
    def _compute_D(self, W):
        D = np.zeros(shape=(W.shape[0], W.shape[0]))
        for i in range(W.shape[0]):
            D[i, i] = np.sum(W[:, i])
        return D
    
    
    def fit_predict(self, X):
        predict = np.zeros(shape=(X.shape[0], self.m))
        self.construct_graph(X)
        for connected_component in self.connected_components():
            W_comp = np.zeros(shape=(len(connected_component), len(connected_component)))
            for ind_i, i in enumerate(connected_component):
                for ind_j, j in enumerate(connected_component):
                    W_comp[ind_i, ind_j] = self.W[i, j]
            D = self._compute_D(W_comp)
            L = D - W_comp
            eig_system = eig(inv(D) @ L)
            eig_vectors, eig_vals = eig_system[1], eig_system[0]
            eig_vectors = eig_vectors[np.argpartition(eig_vals, self.m + 2)][1 : self.m + 1, :]
            for ind_i, i in enumerate(connected_component):
                predict[i , :] = eig_vectors[:, ind_i]
        return predict
