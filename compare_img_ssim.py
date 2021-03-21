# import the necessary packages
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


def mse(imageA, imageB):
    # 均方误差方法用 python实现
    # 计算俩张图片的像素差的平方和的平均值
    # 俩张图必须有相同的 分辨率维度
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB, title):
    # 计算俩张图片的均方误差 及 结构相似性指数
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    # 设置图片的名称标头
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
    # 展示第一张图
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap=plt.cm.gray)
    plt.axis("off")
    # 展示第二张图
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap=plt.cm.gray)
    plt.axis("off")
    # 展示图片
    plt.show()


def main():
    # 加载图片 —— 初始，对比，ps操作过的
    original = cv2.imread("D:/pyimagesearch/images/origin.jpg")
    contrast = cv2.imread("D:/pyimagesearch/images/contrast.jpg")
    shopped = cv2.imread("D:/pyimagesearch/images/photoshopped.jpg")
    # convert the images to grayscale
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

    # 初始化图表
    fig = plt.figure("Images")
    images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)
    # 循环遍历三张图片
    for (i, (name, image)) in enumerate(images):
        # 加载入图片
        ax = fig.add_subplot(1, 3, i + 1)
        ax.set_title(name)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis("off")
    # 展示图片
    plt.show()
    # 分别比较图片
    compare_images(original, original, "Original vs. Original")
    compare_images(original, contrast, "Original vs. Contrast")
    compare_images(original, shopped, "Original vs. Photoshopped")


if __name__ == "__main__":
    main()

