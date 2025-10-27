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
        
        orig_img = cv2.imread(path)
        img = orig_img.copy()
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        max_saturation = np.max(hsv_image[:,:,1])
        color_thresh = 0.8*max_saturation
        white_thresh = 0.2*max_saturation
        #hsv_image[(hsv_image[:,:,1]<color_thresh) & (hsv_image[:,:,1]>white_thresh)]=0
        hsv_image[hsv_image[:,:,1]<color_thresh]=0
        # Limit to red objects
        hsv_image[(hsv_image[:,:,0]>63) & (hsv_image[:,:,0]<200)] = 0

        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        img = cv2.GaussianBlur(img, (75,75),5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img[img>0]=255

        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours and draw bounding boxes
        for contour in contours:
            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
    
            # Draw the rectangle on the original image
            cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green color, 2 pixels thick

        #cv2.imwrite("data/processed/colors/red/road"+str(i)+"_r.png",orig_img)
        #continue


        img = (orig_img[:,:,0]).copy()	
        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Thresholds for binarization
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # Filter by Area
        params.filterByArea = True
        params.minArea = 75 # ~ pi*5^2
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by Convexity
        params.filterByConvexity = True
        #params.minConvexity = 0.87
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(img)
        
        # Draw blobs as green circles
        output = cv2.drawKeypoints(orig_img, keypoints, np.array([]), (255, 0, 0),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imwrite("data/processed/colors/red/road"+str(i)+"_r.png",output)
        continue

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