import numpy as np
import matplotlib.pyplot as plt

class DiscreteModel:
    def __init__(self, v, a1, a2, b, q, t0, k0, x0):
        self.v = v
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.q = q
        self.T0 = t0
        self.k0 = k0
        self.iter_num = round(3 * k0 / t0)
        self.collect_y = []
        self.x0 = np.array([[x0], [x0], [x0]])
        self.A = np.matrix([[0, 1, 0], [0, 0, 1], [-1, -self.a1, -self.a2]])
        self.B = np.array([[0], [0], [self.b]])
        self.C = np.array([1, 0, 0])

    @property
    def count_phi(self):
        Phi = np.zeros((3, 3))
        for q in range(self.q + 1):
            Phi += ((self.A * self.T0) ** q) / np.math.factorial(q)
        return Phi

    @property
    def count_g(self):
        G = np.zeros((3, 3))
        for q in range(self.q):
            G += ((self.A * self.T0) ** q) / np.math.factorial((q + 1))
        G *= self.T0
        G = G.dot(self.B)
        return G

    def count_u(self, iter_n):
        if self.v == 1:
            return np.array([[1]])
        if self.v == 2:
            if iter_n < self.iter_num / 2:
                return np.array([[1]])
            else:
                return np.array([[-1]])
        if self.v == 3:
            if iter_n < self.iter_num / 3 or iter_n > 2 * (self.iter_num / 3):
                return np.array([[1]])
            else:
                return np.array([[-1]])

    def load_collect_y(self):
        x = self.x0
        for i in range(0, self.iter_num, 1):
            y = self.C.dot(x)
            self.collect_y.append(y)
            u_next = self.count_u(i)
            x = self.count_phi.dot(x) + self.count_g.dot(u_next)

    def print_data(self):
        print("Variant: ", self.v)
        print("a1: ", self.a1)
        print("a2: ", self.a2)
        print("b: ", self.b)
        print("T0: ", self.T0)
        print("q: ", self.q)
        print("k: ", self.k0)
        print(self.iter_num)

    def visualize_function(self):
        self.load_collect_y()
        graph_y = []
        for i in range(0, len(self.collect_y), 1):
            graph_y.append(self.collect_y[i].item(0))

        graph_x = []
        for i in range(0, len(graph_y), 1):
            if i >= 1:
                x_prev = graph_x[i - 1]
            else:
                x_prev = 0
            graph_x.append(x_prev + self.T0 * len(graph_y) / 10000)
        plt.plot(graph_x, graph_y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Graph of the original process")
        plt.show()

#v, a1, a2, b, q, t0, k0, x0
res = DiscreteModel(1, 1, 0.9, 1, 1, 0.1, 300, 0.)
res.print_data()
res.visualize_function()