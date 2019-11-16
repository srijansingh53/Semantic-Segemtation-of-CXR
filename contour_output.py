import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('data/outputs/masks_only/IM-0337-0001.jpeg')
org_img = cv2.imread('data/test/images/IM-0337-0001.jpeg')
org_img = cv2.resize(org_img, (256,256))
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
ret,thresh = cv2.threshold(gray_image,127,255,0) 
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(contours, key=cv2.contourArea, reverse=True)
#area = cv2.contourArea(cnt[1])
cnt = [cnt[0],cnt[1]]  
with_contours = cv2.drawContours(org_img,cnt,-1,(0,255,0),3) 
#plt.imshow(with_contours)
cv2.imwrite('documentation/contour.png', with_contours)
