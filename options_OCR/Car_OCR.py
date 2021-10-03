import cv2
import numpy as np
from imutils.contours import sort_contours

def cv_show(name , img, switch = True):
    if switch :
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

# 正确答案
rightAnswer = {0:1, 1:4, 2:0, 3:3, 4:1}

# 读图片
img = cv2.imread('images/1.jpg')

cv_show('image_origin',img)
# 下边填充好判断轮廓
img1 = cv2.copyMakeBorder(img,0,3,0,0,cv2.BORDER_CONSTANT)
# 转换灰度图
img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

##### 摆正试卷 #####
# 高斯滤波 相较于未滤波，后续边缘检测操作得出的轮廓线条会更连续
img_gray_blur = cv2.GaussianBlur(img_gray,(5,5),0)
# 边缘检测
img_edged = cv2.Canny(img_gray_blur,100,200)
# 轮廓检测，并将按闭合周长大小排序后的2到5个轮廓取出
img_edged_contours = cv2.findContours(img_edged.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
img_edged_contours = sorted(img_edged_contours,key = lambda x : cv2.arcLength(x, True), reverse = True)[1:5] 
# 遍历轮廓
for i in img_edged_contours:
    approx = cv2.approxPolyDP(i, 0.02*cv2.arcLength(i,True), True)
    if len(approx) == 4:
        break 
# 透视变换   
img_warped = four_point_transform(img, approx.reshape(4, 2))
img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)


##### 找到选项位置 #####
# 阈值处理
img_thresh = cv2.threshold(img_warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 轮廓检测
img_thresh_contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
# 存放选项轮廓
quartionContours = []
# 选项轮廓筛选
for thisContour in img_thresh_contours :
    (x, y, w, h) = cv2.boundingRect(thisContour)
    # 宽高比
    ar = w/float(h)
    if ar < 1.1 and ar > 0.9 and   w>27 and w<37 and h>27 and h<37 :
        quartionContours.append(thisContour)


##### 比较选项，输出分数
# 轮廓从上到下排序，每行轮廓集中        
quartionContours = sort_contours(quartionContours, method="top-to-bottom")[0]
# 记录得分
score = 0
# 遍历每5个轮廓
for (q, i) in enumerate(np.arange(0, len(quartionContours), 5)):
    # 选取的每行轮廓左->右排序
    cnts = sort_contours(quartionContours[i:i+5])[0]
    # 记录非0值像素个数及轮廓序号
    bubbled = (0,0)
    # 遍历每一个结果
    for (j,c) in enumerate(cnts):
        # 使用mask判断结果
        mask = np.zeros(img_thresh.shape, dtype='uint8')
        # 轮廓内部白色填充
        cv2.drawContours(mask,[c],-1,255,-1)
        # and操作，除选项全变黑
        mask = cv2.bitwise_and(img_thresh,img_thresh,mask=mask)
        # 计算非零值像素个数
        countNonzero = cv2.countNonZero(mask)
        # 判断大小，并取大值
        if countNonzero>bubbled[0]: bubbled = (countNonzero,j)

    # 判断选项正确
    if rightAnswer[q] == bubbled[1]:
        cv2.drawContours(img_warped,[cnts[bubbled[1]]],-1,(0,255,0),3)
        score += 20
    else :
        cv2.drawContours(img_warped,[cnts[bubbled[1]]],-1,(0,0,255),3)

# 打印得分
cv2.putText(img_warped,f'Accuracy rate: {score}%',(10,30),cv2.FONT_HERSHEY_SIMPLEX, \
    0.9, (0,0,255),2)

cv_show('img_warped',img_warped)