import numpy as np
import cvxpy as cp
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from numpy.linalg import eigvals

# Generate synthetic data
X, y = make_blobs(n_samples=200, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)
y = y.reshape(-1,1)
n_samples, n_features = X.shape
def plot_data(X, y):
    plt.figure(figsize=(10, 8))
    
    # Plot each class with different markers and colors
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='b', marker='o', label='Class 1')
    plt.scatter(X[y.ravel() == -1, 0], X[y.ravel() == -1, 1], c='r', marker='s', label='Class -1')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Data Points by Class')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot the data
plot_data(X, y)
def linear_kernel(X):
    return X @ X.T

def polynomial_kernel(X,degree,coef0):
    return (linear_kernel(X)+coef0)**degree

def rbf_kernel(X,gamma):
    X1_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X**2, axis=1).reshape(1, -1)
    distances = X1_norm + X2_norm - 2 * np.dot(X, X.T)
    return np.exp(-gamma*distances)

def sigmoid_kernel(X,alpha,coef0):
    return np.tanh(alpha * linear_kernel(X)+coef0)

def Gram_matrix(X, kernel='linear'):
    if kernel == 'linear':
        return linear_kernel(X)
    elif kernel == 'polynomial':
        return polynomial_kernel(X, degree=3, coef0=1)
    elif kernel == 'rbf':
        return rbf_kernel(X, gamma=1.0)
    elif kernel == 'sigmoid':
        return sigmoid_kernel(X, alpha=0.1, coef0=1)
    else:
        raise ValueError("Unknown kernel: " + kernel)
    
def opti_alpha(X,y):
    alpha = cp.Variable((n_samples,1))  # Lagrange multipliers

    # Compute the Gram matrix
    K = Gram_matrix(X,kernel='linear')

    # Create y*y^T matrix
    yy = y @ y.T

    # Element-wise multiply with kernel matrix
    P = np.multiply(K, yy)

    # Make P positive semidefinite by adding a small regularization term
    # and symmetrizing the matrix
    P = (P + P.T) / 2  # Ensure symmetry
    min_eig = np.min(eigvals(P))
    if min_eig < 0:
        # Add a bit more than the absolute minimum eigenvalue to ensure stability
        P = P + (-min_eig + 1e-8) * np.eye(P.shape[0])

    # Additional numerical stability check
    P = (P + P.T) / 2  # Ensure symmetry again after modification
    epsilon = 1e-10
    P = P + epsilon * np.eye(P.shape[0])  # Add small diagonal term for numerical stability

    # Verify P is positive semidefinite
    min_eig_final = np.min(eigvals(P))
    print(f"Minimum eigenvalue after conditioning: {min_eig_final}")

    # Create the vector of ones
    ones = np.ones((n_samples, 1))

    # Objective function: maximize dual form
    obj = cp.Maximize(ones.T @ alpha - 0.5 * cp.quad_form(alpha, P))

    constraints = [alpha >= 0,
                   alpha <= 1,
                   y.T @ alpha == 0]

    # Solve the optimization problem
    problem = cp.Problem(obj, constraints)
    result = problem.solve(solver=cp.OSQP,polish=1,eps_abs=1e-4,max_iter=20000,verbose=True)
    print(f"Optimization status: {problem.status}")
    print(f"Optimal objective value: {result:.4f}")

    return alpha 

def svm(X,y):
    alpha = opti_alpha(X,y)
    sv_threshold = 1e-4
    sv_indices = np.where(alpha.value > sv_threshold)[0]

    # Compute w and b
    w = np.sum(alpha.value[sv_indices] * y[sv_indices] * X[sv_indices], axis=0)
    support_vectors = X[sv_indices]
    support_vector_y = y[sv_indices]

    # Compute b using average over support vectors
    margin_vectors = []
    for i in sv_indices:
        # Compute decision value for each support vector
        decision_value = np.dot(X[i], w)
        margin_vectors.append(y[i] - decision_value)
    b = np.mean(margin_vectors)

    return w,b,support_vectors,sv_indices

w,b,support_vectors,sv_indices = svm(X,y)
# Plotting
def plot_svm_results():
    plt.figure(figsize=(10, 8))
    
    # Plot all points
    plt.scatter(X[y.ravel()==1, 0], X[y.ravel()==1, 1], c='b', marker='o', label='Class 1')
    plt.scatter(X[y.ravel()==-1, 0], X[y.ravel()==-1, 1], c='r', marker='s', label='Class -1')
    
    # Plot support vectors
    plt.scatter(X[sv_indices, 0], X[sv_indices, 1], s=200, linewidth=1, 
               facecolors='none', edgecolors='g', label='Support Vectors')
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    Z = (xx.ravel() * w[0] + yy.ravel() * w[1] + b).reshape(xx.shape)
    
    plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='-')
    plt.contour(xx, yy, Z, levels=[-1, 1], colors='gray', linestyles='--')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary with Support Vectors')
    plt.legend()
    plt.show()

# Print results

print(f"Number of support vectors: {len(sv_indices)}")


# Plot results
plot_svm_results()


# Print some additional information about the solution
print(f"\nModel Parameters:")
print(f"w: {w}")
print(f"b: {b}")
print(f"Number of support vectors: {len(sv_indices)} out of {n_samples} points")