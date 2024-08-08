import numpy as np
import control as ctrl

# Define the system matrices
A = np.array([[-2, -1, 2],
              [-1, -2, 2],
              [-2, 0, 2]])

B = np.array([[0, 0],
              [0, 1],
              [1, 0]])

f = np.array([[1],
              [1]])

# Calculate b = B*f
b = B @ f

# Compute the controllability matrix
C = ctrl.ctrb(A, b)

# Given matrices
Psi = np.array([[1, 2, -1],
                [0, 1, 2],
                [0, 0, 1]])

delta = np.array([4, 13, 10])

# Compute M matrix
M = delta @ np.linalg.inv(C @ Psi)

# Compute K1
K1 = f * M

# Desired closed-loop poles
pd = np.array([-2, -2, -2])

# Compute the state feedback gain using pole placement
k = ctrl.acker(A, B, pd)

# Compute K2
K2 = f * k

# Compute the closed-loop system matrix Ac
Ac = A - B @ K1

# Compute eigenvalues of Ac
eigvals_Ac = np.linalg.eigvals(Ac)

print("State feedback gain K1:")
print(K1)
print("\nState feedback gain k (pd = [-2, -2, -2]):")
print(k)
print("\nState feedback gain K2:")
print(K2)
print("\nEigenvalues of Ac:")
print(eigvals_Ac)
