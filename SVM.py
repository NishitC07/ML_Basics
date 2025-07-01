import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Original 4D dataset
X = np.array([
    [25, 1, 1, 0],  # sample 1
    [30, 2, 0, 1],  # sample 2
    [35, 1, 0, 0],  # sample 3
    [40, 0, 1, 1],  # sample 4
    [27, 0, 1, 1],  # sample 5
    [21, 1, 1, 1],  # sample 6
    [30, 2, 0, 1],  # sample 7
    [32, 0, 0, 0],  # sample 8
])

y = np.array([0, 0, 0, 1, 1, 1, 0, 0])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce 4D to 2D using PCA for visualization
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_scaled)
print(X_vis.shape, X.shape)
print(X_vis)
# Different C values to observe decision boundary changes
C_values = [0.1, 1, 10, 100]

# Plotting setup
fig, axes = plt.subplots(1, len(C_values), figsize=(20, 5))

for i, C in enumerate(C_values):
    clf = SVC(kernel='rbf', C=C)
    clf.fit(X_vis, y)

    ax = axes[i]
    ax.set_title(f"SVM Decision Boundary (C={C})")

    # Create mesh grid for contour plot
    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.decision_function(grid_points).reshape(xx.shape)

    # Plot decision regions and boundary
    ax.contourf(xx, yy, Z > 0, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'], colors='k')
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=60)

plt.tight_layout()
plt.show()
