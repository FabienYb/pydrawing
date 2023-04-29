import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法找到图像文件：{image_path}")
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def detect_edges(grayscale_image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(grayscale_image, low_threshold, high_threshold)
    return edges

def extract_points(edges, sampling_rate=0.05):
    y, x = np.where(edges > 0)
    points = np.column_stack((x, y))
    num_points = len(points)
    num_sampled_points = int(sampling_rate * num_points)
    sampled_points = points[np.random.choice(num_points, num_sampled_points, replace=False)]
    return sampled_points

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return intersection_x, intersection_y
    return None

def count_intersections(lines, new_line):
    intersections = 0
    for line in lines:
        intersection = line_intersection(line, new_line)
        if intersection is not None:
            intersections += 1
    return intersections

def animate(i, points, lines, ax, max_overlaps):
    if i > 0:
        new_line = [points[i - 1, 0], points[i - 1, 1], points[i, 0], points[i, 1]]
        intersections = count_intersections(lines, new_line)
        alpha = 1 - (intersections / max_overlaps)
        line = Line2D(new_line[:2], new_line[2:], lw=1, color='black', alpha=alpha)
        ax.add_line(line)
        lines.append(new_line)
    return ax.lines,

def plot_image_from_edges(edges, interval=5, max_overlaps=10):
    fig, ax = plt.subplots(figsize=(10, 10))
    points = extract_points(edges)
    num_points = len(points)
    ax.set_xlim(0, edges.shape[1])
    ax.set_ylim(edges.shape[0], 0)
    ax.axis('off')
    lines = []
    ani = FuncAnimation(fig, animate, frames=num_points - 1, fargs=(points, lines, ax, max_overlaps), interval=interval, blit=False)
    plt.show()

def main(image_path):
    grayscale_image = read_image(image_path)
    edges = detect_edges(grayscale_image)
    plot_image_from_edges(edges)

if __name__ == "__main__":
    image_path = '3.jpg'
    main(image_path)