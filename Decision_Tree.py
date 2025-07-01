from sklearn.tree import DecisionTreeClassifier
import numpy as np
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

model = DecisionTreeClassifier()
model.fit(X, y)

from sklearn.tree import export_text
print(export_text(model, feature_names=["Age", "Income", "Student", "Credit"]))

print(model.predict(np.array([[1,1,0,1]])))