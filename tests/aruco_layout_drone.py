import matplotlib.pyplot as plt
import numpy as np


def plot_square(ax, position, rotation_matrix, size):
    square = np.array([
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0]
    ]) * size / 2

    rotated_square = square.dot(rotation_matrix.T) + position

    X = rotated_square[:, 0].reshape((2, 2))
    Y = rotated_square[:, 1].reshape((2, 2))
    Z = rotated_square[:, 2].reshape((2, 2))

    ax.plot_surface(X, Y, Z, alpha=0.5)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

rotations_and_positions = [
    (np.eye(3), [1, 0, 0]),
    (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), [0, 1, 0]),
    (np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]), [0, 0, 1]),
    (np.array([[0.7071, 0.7071, 0], [-0.7071, 0.7071, 0], [0, 0, 1]]), [-1, -1, -1])
]

for i, (rotation, position) in enumerate(rotations_and_positions):
    plot_square(ax, np.array(position), rotation, 0.2)
    ax.text(position[0], position[1], position[2], f'Marker {i + 1}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()