import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

def extract_contours(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def fourier_transform(contour):
    complex_contour = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    fourier_coefficients = np.fft.fft(complex_contour)
    return fourier_coefficients

def fourier_epicycles(coefficients, time, num_coefficients=None):
    if num_coefficients is None:
        num_coefficients = len(coefficients)
    else:
        num_coefficients = min(num_coefficients, len(coefficients))

    omega = 2 * np.pi * time
    epicycles = []
    pos = 0

    for n in range(num_coefficients):
        radius = abs(coefficients[n])
        angle = np.angle(coefficients[n]) + n * omega
        next_pos = pos + radius * np.exp(1j * angle)
        epicycles.append((pos, radius, angle))
        pos = next_pos

    return epicycles

def animate_drawing(image_path, detail_level=1):
    image = imageio.imread(image_path)
    contours = extract_contours(image)
    if not contours:
        print("No contours found. Please check the image path and try again.")
        return

    largest_contour = max(contours, key=lambda x: x.shape[0])
    fourier_coefficients = fourier_transform(largest_contour)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=3)

    def init():
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        return line,

    def update(frame):
        ax.clear()
        num_coefficients = 1 + frame // (10 // detail_level)
        time = frame / 200
        epicycles = fourier_epicycles(fourier_coefficients, time, num_coefficients)
        path = np.array([c[0] for c in epicycles])

        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])

        for i in range(len(epicycles)):
            pos, radius, angle = epicycles[i]
            circle = plt.Circle((pos.real, pos.imag), radius, color='r', fill=False)
            ax.add_artist(circle)
            if i < len(epicycles) - 1:
                ax.plot([pos.real, path[i + 1, 0]], [pos.imag, path[i + 1, 1]], 'r--')

        line, = ax.plot(path[:, 0], path[:, 1], lw=3)
        return line,

    ani = FuncAnimation(fig, update, frames=range(0, 200 * detail_level), init_func=init, blit=False, interval=50)
    plt.gca().invert_yaxis()
    plt.show()

# 使用您的图像路径


# 使用您的图像路径替换下面的 'your_image_path.jpg'
# detail_level 是一个正整数，数值越大，边框越精细，计算量也
# 使用您的图像路径替换下面的 'your_image_path.jpg'
# detail_level 是一个正整数，数值越大，边框越精细，计算量也会增加
animate_drawing('2.jpg', detail_level=2)
