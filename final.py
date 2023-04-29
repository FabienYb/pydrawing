import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift

def read_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def resize_image(image_array, new_size):
    resized_img = cv2.resize(image_array, new_size, interpolation=cv2.INTER_AREA)
    return resized_img

def fourier_transform(image_array):
    f_transform = fft2(image_array)
    f_shift = fftshift(f_transform)
    return f_shift

def inverse_fourier_transform(f_shift):
    f_ishift = ifftshift(f_shift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def visualize_fourier_drawing(image_path, new_size=(164, 164)):
    image_array = read_image(image_path)
    resized_image_array = resize_image(image_array, new_size)

    f_shift = fourier_transform(resized_image_array)

    num_of_rows = resized_image_array.shape[0]
    fig, ax = plt.subplots()
    for i in range(1, num_of_rows + 1):
        f_shift_partial = np.zeros_like(f_shift, dtype=complex)
        f_shift_partial[:i, :] = f_shift[:i, :]

        img_back = inverse_fourier_transform(f_shift_partial)

        ax.clear()
        ax.imshow(img_back, cmap='gray', vmin=0, vmax=255)
        plt.pause(0.05)

    plt.show()

if __name__ == '__main__':
    image_path = '1.jpg'
    visualize_fourier_drawing(image_path)
