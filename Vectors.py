# # Day 1

# import numpy as np
# import matplotlib.pyplot as plt

# # Define vectors
# v1 = np.array([1, -1])
# v2 = np.array([3, -3])
# # Vector magnitude
# magnitude_v1 = np.linalg.norm(v1)

# # Dot product
# dot = np.dot(v1, v2)

# # Matrix multiplication
# A = np.array([[1, 2, 3], [3, 4, 5]])#shape (2,3) - outer -> inner counting
# B = np.array([[2, 0], [1, 3],[8,4]])

# print(v1.shape)
# C = np.matmul(A, B)
# D = A @ B 

# print("v1 magnitude:", magnitude_v1)
# print("Dot product of v1 and v2:", dot)
# print("Matrix product:\n", C)
# print(D)


# dot_product = np.dot(v1, v2)

# # Norms (magnitudes)
# norm_a = np.linalg.norm(v1)
# norm_b = np.linalg.norm(v2)

# # Cosine similarity
# cos_sim = dot_product / (norm_a * norm_b)

# print("Dot Product:", dot_product)
# print("Cosine Similarity:", cos_sim)

# plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r')
# plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b')
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
# plt.grid()
# plt.title("Vector Visualization")
# plt.show()

# # Day 2

# import numpy as np
# import matplotlib.pyplot as plt

# def plot_vector_transformation(matrix, title):
#     original = np.array([[1, 0], [0, 1]])  # Unit vectors i, j
#     transformed = matrix @ original

#     plt.figure()
#     plt.quiver(*np.zeros((2, 2)), original[0], original[1], color=['r', 'b'], scale=1, angles='xy', scale_units='xy', label='Original')
#     plt.quiver(*np.zeros((2, 2)), transformed[0], transformed[1], color=['orange', 'green'], scale=1, angles='xy', scale_units='xy', label='Transformed')
#     plt.xlim(-3, 3)
#     plt.ylim(-3, 3)
#     plt.axhline(0, color='gray')
#     plt.axvline(0, color='gray')
#     plt.grid()
#     plt.title(title)
#     plt.legend(['i', 'j'])
#     plt.show()

# # Try different matrices
# rotation_90 = np.array([[0, -1], [1, 0]])
# scaling = np.array([[2, 0], [0, 1.5]])
# shear_x = np.array([[1, 1], [0, 1]])

# plot_vector_transformation(rotation_90, "90° Rotation")
# plot_vector_transformation(scaling, "Scaling")
# plot_vector_transformation(shear_x, "Shear in X direction")


# # Day-3

# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
# docs = ["AI is amazing isnt it?", "Machine learning is part of AI","we'll glowing in the dark"]
# tfidf = TfidfVectorizer()
# features = tfidf.fit_transform(docs).toarray()
# print(tfidf.get_feature_names_out())
# features = [[np.array([1,3])],[np.array([2,1])]]
# print(features)
# print("dot", np.dot(features[0][0],features[1][0]))
# print(cosine_similarity([features[0][0]], [features[1][0]]))
# print("Cosine Similarity:", cosine_similarity([features[0][0]], [features[1][0]])[0][0])


import numpy as np
import matplotlib.pyplot as plt

# Generate dummy data
X = np.linspace(0, 10, 100)
Y = 3 * X + 7 + np.random.randn(100) *2   # true slope=3, bias=7
print(X)
print(Y)
# Initialize parameters
w = 0
b = 0
lr = 0.01
epochs = 10000
n = len(X)

# Training using gradient descent
for epoch in range(epochs):
    Y_pred = w * X + b
    error = Y_pred - Y
    
    # Compute gradients
    dw = (2/n) * np.dot(error, X)
    db = (2/n) * np.sum(error)

    # Update weights
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch}: Loss = {loss:.2f}, w = {w:.2f}, b = {b:.2f}, error = {error}  " )

# Final predictions
plt.scatter(X, Y, label='Data')
plt.plot(X, w * X + b, color='red', label='Fitted Line')
plt.legend()
plt.title("Linear Regression From Scratch")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

