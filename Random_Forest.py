from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

X = np.array([
    [25, 1, 1, 0],  # sample 1
    [30, 2, 0, 1],  # sample 2
    [35, 1, 0, 0],  # sample 3
    [40, 0, 1, 1],  # sample 4
    [27, 0, 1, 1],
    [21, 1, 1, 1],  
    [30, 1, 0, 1],  
    [32, 0, 0, 0],  # sample 5
])

y = np.array([0, 0, 0, 1, 1, 1, 0, 0])
feature_names=["Age", "Income", "Student", "Credit"]

# Train a Random Forest
model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
model.fit(X, y)

# Extract and print tree 0
for i, tree in enumerate(model.estimators_):
    print(f"\n--- Tree {i} ---")
    print(export_text(tree, feature_names=feature_names))

plt.figure(figsize=(12, 8))
plot_tree(model.estimators_[0], feature_names=feature_names, filled=True)
plt.show()