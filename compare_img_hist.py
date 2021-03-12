import cv2 as cv

def compare_img_hist(img1, img2):
    """
    Compare the similarity of two pictures using histogram(直方图)
        Attention: this is a comparision of similarity, using histogram to calculate

        For example:
         1. img1 and img2 are both 720P .PNG file,
            and if compare with img1, img2 only add a black dot(about 9*9px),
            the result will be 0.999999999953

    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: the similarity of two pictures
    """
    # Get the histogram data of image 1, then using normalize the picture for better compare
    img1_hist = cv.calcHist([img1], [1], None, [256], [0, 256])
    img1_hist = cv.normalize(img1_hist, img1_hist, 0, 1, cv.NORM_MINMAX, -1)


    img2_hist = cv.calcHist([img2], [1], None, [256], [0, 256])
    img2_hist = cv.normalize(img2_hist, img2_hist, 0, 1, cv.NORM_MINMAX, -1)

    similarity = cv.compareHist(img1_hist, img2_hist, 0)

    return similarity


# 将图片转化为RGB
def make_regalur_image(img, size=(64, 64)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image


# 计算直方图
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist


# 计算相似度
def calc_similar(li, ri):
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim
