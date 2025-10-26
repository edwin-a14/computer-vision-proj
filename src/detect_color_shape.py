import cv2
import numpy as np
import os

def main():
    # try and detect circular red things to check first
    for i in range(877):
        #if i<300 or i>500:
            #continue
        
        path = "data/raw/kaggle_roadsign/images/road"+str(i)+".png"
        #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        
        img = cv2.imread(path)
        r = img[:,:,2]
        g = img[:,:,1]
        b = img[:,:,0]
        img[(g>=r)|(b>=r)]=0
        white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #white[white<200]=0

        max_red = np.max(r)
        h, w = np.shape(r)
        if (max_red!=0):
            for k in range(h):
                for j in range(w):
                    r[k,j]=int(r[k,j]*255./max_red)
        
        #r[white>200]=white[white>200]

        img = r
        #cv2.imwrite("data/processed/colors/red/road"+str(i)+"_r.png",img)
        #continue

        """
        #img = cv2.Canny(r,0.6, 0.9)
        circles = cv2.HoughCircles(r.astype(np.uint8),cv2.HOUGH_GRADIENT, 2, 16, 0.9, 0.95)
        """
        
        img = cv2.GaussianBlur(img,(9,9),2)
        img = cv2.Canny(img,0.9, 0.95)

        #cv2.imwrite("data/processed/colors/red/road"+str(i)+"_r.png",img)
        #continue

        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0.
        valid = []
        img = cv2.imread(path)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True) # True for closed contour

            # Avoid division by zero for contours with zero perimeter
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                if circularity>0.6:
                    max_area=max(max_area,area)
                    valid.append(contour)
        if len(valid)==0:
            continue
        
        for contour in valid:
            if cv2.contourArea(contour)>=0.9*max_area:
                img=cv2.drawContours(img,[contour],0,(0,255,0),1)

        """
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 2, 16, 0.9, 0.95)
        shape = np.shape(circles)
        if shape[0]==4:
            continue
        circles = circles[0]
        img = cv2.imread(path)
        for circle in circles[:]:
            img = cv2.circle(img,(int(circle[0]),int(circle[1])),int(circle[2]),(0, 255, 0), 1)
        """
        cv2.imwrite("data/processed/colors/red/road"+str(i)+"_r.png",img)
        
# detect vertical lines in images -- signs in proximity
# cv.HoughLines

if __name__ == "__main__":
    main()