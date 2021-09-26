from typing import Counter
import cv2 
from imutils import contours as ctrs
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.shape_base import dstack

#一个简单的图片显示函数 
def cv_show(name , img):
    #图像的显示，也可以创建多个窗口
    cv2.imshow(name,img)
    #等待时间 ms级 0代表任意键终止
    cv2.waitKey(0)
    #关闭所有窗口
    cv2.destroyAllWindows()

# 读取图片 灰度图
img         = cv2.imread('image/cart1.jpg',0)
img_oringe  = cv2.imread('image/cart1.jpg')
template0   = cv2.imread('image/number.jpg',0)
template    = cv2.imread('image/number.jpg')

###################################### 对模板的处理 ################################################

# 转换为二值图片
Binary_template = cv2.threshold(template0, 10 , 255 , cv2.THRESH_BINARY_INV)[1]
# 获取轮廓参数
binary, contours, hierarchy = cv2.findContours(Binary_template.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 画出轮廓
template_contours = cv2.drawContours(template.copy(),contours,-1,(0, 0, 255),2)
# 轮廓排序  将轮廓从左到右依次排序
contours = ctrs.sort_contours(contours,method ='left-to-right')[0] 

# 创建数字模板字典 每个索引对应一个数字的模板 
digits = {}
# 枚举每一个轮廓
for i,n in enumerate(contours):
    # 得到轮廓的外接矩形坐标
    x, y, w, h = cv2.boundingRect(n)
    # 在原图上切出对应矩形位置
    roi = template0[y:y+h, x:x+w]
    # 大小重置
    roi = cv2.resize(roi,(57,88))
    # cv_show('roi', roi)
    # 载入字典 每个数字对应一个模板
    digits[i] = roi


################################################ 原图片处理 ################################################

# 初始化卷积核
rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT,(12,3))


# 读取输入图像，预处理为宽高比不变 宽变为width的图片
width = 400
img_shape = (width, int(width/img.shape[1]*img.shape[0]))
img = cv2.resize(img,img_shape)
img_oringe = cv2.resize(img_oringe,img_shape)

# 形态学礼帽操作 突出明亮区域
img = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,rectkernel)

# sobel算子的梯度计算
img_SobelX = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3) 
img_SobelX = cv2.convertScaleAbs(img_SobelX)
 
img_SobelY = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
img_SobelY = cv2.convertScaleAbs(img_SobelY)

img = cv2.addWeighted(img_SobelX,0.5,img_SobelY,0.5,0)


###### 获取轮廓 #########
# 形态学闭操作 先膨胀后腐蚀，获取数字块
img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE,sqkernel)

# 阈值处理 THRESH_OTSU会自动寻找合适的阈值，适合双峰，需要把阈值参数设为0
img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv_show('thresh Image',thresh)

# 获取大致轮廓
img1_contours = cv2.findContours(img1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
img2 = cv2.drawContours(img_oringe.copy(),img1_contours,-1,(0,0,255),2)


locs = []

# 遍历轮廓 寻找合适的区域轮廓
for i,n in enumerate(img1_contours):
    # 画出每个轮廓的外接矩形
    (x,y,w,h) = cv2.boundingRect(n)
    # 宽高比
    ar = w/float(h)
    # 根据宽高比，选择合适的区域
    if ar > 2.5 and ar < 3.5 :
        if (w < 75 and w > 60) and ( h > 20 and h < 35) :
            locs.append((x,y,w,h))
# 轮廓位置排序
locs.sort(key= lambda x:x[0])

for (x, y, w, h) in locs :
    cv2.rectangle(img_oringe,(x,y),(x+w,y+h),(0,0,255),2)

# 输出列表
groupOutput = []

##################################### 模板匹配： ########################
for (i,((x, y, w, h))) in enumerate(locs):

    # 将每一个轮廓往外扩一点点 以便判断
    group = img[y-5 : y+h+5, x-5 : x+w+5]
    # 阈值处理 二值化
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 获取轮廓信息
    group_contours = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    # 轮廓排序
    group_contours = ctrs.sort_contours(group_contours,method ='left-to-right')[0] 

    # 匹配每一个数字
    for j in group_contours:
        # 找到当前数字的轮廓，resize合适的大小
        (x,y,w,h) = cv2.boundingRect(j)
        # 在每个小轮廓图片上切出对应矩形位置
        roi = group[y:y+h, x:x+w]
        # 大小重置
        roi = cv2.resize(roi,(57,88))
        roi = cv2.threshold(roi.copy(), 10 ,255 ,cv2.THRESH_BINARY_INV)[1]

        # 模板匹配得分
        scores = []

        # 在模板中计算各个数字匹配度
        for digit, digitROI in digits.items():
            # 模板匹配
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))
    groupOutput.append(' ')

print('您的银行卡号为：', ''.join(x for x in groupOutput))
