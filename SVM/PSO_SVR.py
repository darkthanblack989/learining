import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import pyswarms as ps
 
np.random.seed(42)
 
 
# 适应度函数
def fitness_function(params, X, y, xt, yt):
    C, gamma = params
    svm_model = SVR(kernel='rbf', gamma=gamma, C=C)
    svm_model.fit(X, y)
    y_pred = svm_model.predict(xt)
    mse = np.mean((yt - y_pred) ** 2)
    return mse
 
 
# 优化函数
def optimize_svm(X, y, xt, yt, n_particles=100, n_iterations=100):
    def _fitness_function(params):
        fitness_values = []
        for p in params:
            fitness = fitness_function(p, X, y, xt, yt)
            fitness_values.append(fitness)
        return fitness_values
 
    # 参数边界
    bounds = (np.array([0.1, 0.01]), np.array([50.0, 10.0]))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
 
    cost_history = np.zeros(n_iterations)
    # 运行优化器进行参数优化
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=2, bounds=bounds, options=options)
 
    best_cost, best_params = optimizer.optimize(_fitness_function, iters=n_iterations)
 
    # 在每次迭代前保存代价值
    for i, cost in enumerate(optimizer.cost_history):
        cost_history[i] = cost
 
 
    # 根据优化结果建立最终的SVM模型
    C, gamma = best_params
    svm_model = SVR(kernel='rbf', gamma=gamma, C=C)
    svm_model.fit(X, y)
    print('最优参数：', best_params)
    # 绘制代价值变化曲线
    plt.plot(range(n_iterations), cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function Evolution')
    plt.show()
 
    return svm_model