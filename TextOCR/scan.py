from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from numpy.core import numeric
from numpy.lib.shape_base import tile


# 一个简单的图像显示函数
def cv_show(name,scr,waitkey = 0):
    cv2.imshow(str(name),scr)
    cv2.waitKey(waitkey)
    cv2.destroyAllWindows()

# 一个简单的图像按比例重置函数
def resize(img, heigh=None, width=None,  inter=cv2.INTER_AREA):
    if not width and not heigh : return img
    if not width : return cv2.resize(img, (int(heigh/float(img.shape[0])*img.shape[1]), heigh), interpolation=inter)
    if not heigh : return cv2.resize(img, (width, int(width/float(img.shape[1])*img.shape[0])), interpolation=inter)
    return cv2.resize(img, (width, heigh), interpolation=inter)

def order_points(pts):
    # 创建4x2的列表
    rect = np.zeros((4,2),dtype="float32")
    # 压缩列，将每个坐标的x与y相加，
    # 所得最小值则为左上角坐标
    # 所得最大值则为右下角坐标
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算离散值，将每个坐标y与x相减
    # 所得最小值则为右上角坐标
    # 所得最大值则为左下角坐标
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 坐标分布左上-右上-右下-左上
    return rect

def four_point_transform(image,pts):
    # 将输入的坐标转换成左上-右上-右下-左上排列
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算下边和上边的长度，比较长的作为新图像的宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # 计算右边和左边的长度，比较长的作为新图像的高度
    HeightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    HeightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(HeightA), int(HeightB))
    # 确定新图像的顶点坐标，按左上-右上-右下-左上排列
    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype= "float32")
    # 使用该函数来将原图像坐标rect按新坐标dst转换成新图像得到新矩阵
    M = cv2.getPerspectiveTransform(rect,dst)
    # 转换成新的鸟瞰图，参数为：图像、转换矩阵M、输出图片的宽度和高度。
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# 一个简单的图像旋转函数
def  angle_transform(img, degree, Crop=False, color = (0,0,0)):
    h, w = img.shape[:2]
    # 生成旋转矩阵：旋转中心，旋转角度，缩放比例
    M = cv2.getRotationMatrix2D((w/2,h/2),degree,1)
    if(Crop):  
        return cv2.warpAffine(img.copy(),M,(w,h),\
        borderMode=cv2.BORDER_CONSTANT, borderValue=color)
    # 进行旋转变换
    new_w = int(h*np.abs(M[0,1]) + w*np.abs(M[0,0]))
    new_h = int(h*np.abs(M[0,0]) + w*np.abs(M[0,1]))
    # 重新设定旋转中心偏移量
    M[0,2] += new_w/2 - w/2
    M[1,2] += new_h/2 - h/2
    return cv2.warpAffine(img.copy(),M,(new_w,new_h),\
     borderMode=cv2.BORDER_CONSTANT, borderValue=color)

img_origin = cv2.imread('Images/img1.jpg')

ratio = img_origin.shape[0]/500.0
orig = img_origin.copy()

""" # 图像大小按原比例重置
image = resize(orig, heigh = 500) """
image = orig


# 预处理
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# 高斯滤波
gray = cv2.GaussianBlur(image,(7,7),0)
# 边缘检测
edged = cv2.Canny(gray, 75, 100) 

# 图片的所有轮廓存储在一条列表中
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
# 将轮廓按体积从大到小排列 取前5个轮廓
cnts = sorted(cnts,key = lambda x: cv2.arcLength(x, True), reverse=True)[1:5]
# 遍历轮廓
for i in cnts:
    # 计算轮廓近似
    approx = cv2.approxPolyDP(i, 0.02*cv2.arcLength(i,True), True)
     # 4个点的时候就拿出来
    if len(approx) == 4:
        break  
# 将选中轮廓中的图形还原成原来的比例并透视变换
warped = four_point_transform(orig, approx.reshape(4,2))#*ratio)
# 将图像顺时针旋转90°
warped = angle_transform(warped,90)

# 二值处理
img =cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
# 自适应均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
# 应用于图片中
ref = clahe.apply(img)
text = pytesseract.image_to_string(ref)
print(text)



