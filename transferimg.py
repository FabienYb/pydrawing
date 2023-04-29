import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
image_path=input(str())
def read_image(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def detect_edges(grayscale_image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(grayscale_image, low_threshold, high_threshold)
    return edges

def sample_points(edges, sample_rate=0.1):
    y, x = np.where(edges > 0)
    num_points = int(sample_rate * len(x))
    indices = np.random.choice(len(x), num_points, replace=False)
    return x[indices], y[indices]

def animate(i, x, y, ax, edges):
    ax.clear()
    ax.scatter(x[:i], y[:i], s=1, c='black')
    ax.set_xlim(0, edges.shape[1])
    ax.set_ylim(edges.shape[0], 0)
    ax.axis('off')

def plot_image_from_edges(edges, sample_rate=1):
    fig, ax = plt.subplots(figsize=(10, 10))
    x, y = sample_points(edges, sample_rate=sample_rate)
    num_points = len(x)
    ani = FuncAnimation(fig, animate, frames=num_points, fargs=(x, y, ax, edges), interval=0.0001)
    plt.show()


def main(image_path):
    grayscale_image = read_image(image_path)
    edges = detect_edges(grayscale_image)
    plot_image_from_edges(edges)

if __name__ == "__main__":
    image_path = '3.jpg'
    main(image_path)
