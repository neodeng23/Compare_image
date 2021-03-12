from PIL import Image
from PIL import ImageChops


# from PIL import ImageEnhance

def compare_images(path_one, path_two, diff_save_location):
    """
    比较图片，如果有不同则生成展示不同的图片

    @参数一: path_one: 第一张图片的路径
    @参数二: path_two: 第二张图片的路径
    @参数三: diff_save_location: 不同图的保存路径
    """
    image_one = Image.open(path_one)
    image_two = Image.open(path_two)
    try:
        diff = ImageChops.difference(image_one, image_two)
        # diff.show()            #不同的点图
        r, g, b = diff.split()  # RGB分离
        invertr = ImageChops.invert(r)  # 红色反向
        img1 = invertr.convert('1')  # 转成黑白图，黑点即红点

        img = img1.convert("RGBA")  # 转换格式，确保像素包含alpha通道
        width, height = img.size  # 长度和宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = img.getpixel((i, j))  # 获取一个像素
                if (data.count(255) == 4):  # RGBA都是255，改成透明色
                    img.putpixel((i, j), (255, 255, 255, 0))
                else:
                    # if img.getpixel((i,j))[0]>200:
                    img.putpixel((i, j), (255, 0, 0, 255))
                    img.paste((255, 0, 0, 255), (i - 4, j - 4, i, j))  # 放大红点范围
        img = img.convert("RGB")
        image_one_l = image_one.convert("L")  # 转化为灰度
        image_one = image_one_l.convert("RGB")  # 再把灰度图像转为RGB
        # r,g,b=image_one.split()
        r, g, b = img.split()
        image = Image.composite(image_one, img, g)
        image.show()
        image.save("最终_" + diff_save_location)

        if diff.getbbox() is None:
            # 图片间没有任何不同则直接退出
            print("【+】We are the same!")
        else:
            diff.save(diff_save_location)
    except ValueError as e:
        text = ("表示图片大小和box对应的宽度不一致，参考API说明：Pastes another image into this image."
                "The box argument is either a 2-tuple giving the upper left corner, a 4-tuple defining the left, upper, "
                "right, and lower pixel coordinate, or None (same as (0, 0)). If a 4-tuple is given, the size of the pasted "
                "image must match the size of the region.使用2纬的box避免上述问题")
        print("【{0}】{1}".format(e, text))


if __name__ == '__main__':
    compare_images('1.jpg',
                   '2.jpg',
                   '我们不一样.png')
