import cv2
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.cluster import DBSCAN

def main():
    # try and detect circular red things to check first
    for road_sign_num in range(877):
        #if i<300 or i>500:
            #continue
        
        path = "data/raw/kaggle_roadsign/images/road"+str(road_sign_num)+".png"
        #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        
        orig_img = cv2.imread(path)
        img = orig_img.copy()

        # Non-linear blurring preserving edges
        img = cv2.medianBlur(img, 5)

        hsv_image = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

        h = hsv_image[:,:,0]
        """
        r = img[:,:,0]
        img = r
        """

        h = k_means_merge(h, 8)
        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite("data/processed/colors/red/road"+str(road_sign_num)+"_r.png",img)
        continue
        
        """
        # Plot saturation graphs

        saturation_channel = hsv_image[:, :, 1]

        # Calculate the histogram for the saturation channel
        # parameters:
        # - [saturation_channel]: The image (or channel) to compute the histogram for, wrapped in a list.
        # - [0]: The channel index (0 for saturation in this case, as it's a single channel array).
        # - None: No mask is applied.
        # - [256]: The number of bins for the histogram (0-255 for 8-bit images).
        # - [0, 256]: The range of pixel values (saturation values range from 0 to 255).
        hist_saturation = cv2.calcHist([saturation_channel], [0], None, [256], [0, 256])

        # Plot the saturation histogram
        plt.figure(figsize=(8, 6))
        plt.plot(hist_saturation, color='red')
        plt.title('Saturation Histogram')
        plt.xlabel('Saturation Value')
        plt.ylabel('Number of Pixels')
        plt.xlim([0, 256])
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf)
        # Convert PIL image to NumPy array (OpenCV format)
        # Note: Matplotlib saves in RGB, OpenCV expects BGR for color images
        # So, we convert RGB to BGR
        opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 3. Save the image using cv2.imwrite()
        filename = "data/raw/kaggle_roadsign/images/road"+str(road_sign_num)+"_h_diagram.png"
        cv2.imwrite(filename, opencv_img)

        plt.close()
        continue
        """       

        """
        # Threshold by saturation
        max_saturation = np.max(hsv_image[:,:,1])
        #saturation_signal = (hsv_image[:,:,1]).flatten()
        #saturation_signal.sort()
        #saturation_signal = saturation_signal[len(saturation_signal)/2:]
        #saturation_signal *= -1
        peaks, _ = find_peaks(saturation_signal, distance=len(saturation_signal)/2)
        color_thresh = np.quantile(hsv_image[:,:,1],0.75)
        white_thresh = 0.2*max_saturation
        #hsv_image[(hsv_image[:,:,1]<color_thresh) & (hsv_image[:,:,1]>white_thresh)]=0
        hsv_image[hsv_image[:,:,1]<color_thresh]=0
        # Limit to red objects
        #hsv_image[(hsv_image[:,:,0]>47) & (hsv_image[:,:,0]<207)] = 0
        """

        """
        # Blur colors
        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        img = cv2.GaussianBlur(img, (75,75),5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_thresh = 50
        img[img<gray_thresh]=0
        """

        #img = cv2.Canny(img, 0.8, 0.9)
        


        """
        # Find Connected Regions

        h_values = np.unique(h)
        for h_val in h_values:
            
            h_sample = np.zeros_like(h)
            h_sample[h==h_val] = 255
            h_sample = h_sample.astype(np.uint8)
            
            regions = cv2.connectedComponentsWithStats(h_sample, 4, cv2.CV_32S)
            (num_labels, labels, stats, centroids) = regions

            # Loop through each connected component
            for i in range(1, num_labels):
                # Extract the bounding box coordinates
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                # Optional: Filter out small components by area
                if area > 75:
                    # Draw the bounding box on the original image
                    cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    # Add a text label (optional)
                    #cv2.putText(orig_img, "Component " + str(road_sign_num), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        """

        
        # Find contours
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through contours and draw bounding boxes
        for contour in contours:
            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
        
            # Draw the rectangle on the original image
            cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green color, 2 pixels thick    

        #cv2.imwrite("data/processed/colors/red/road"+str(road_sign_num)+"_r.png",orig_img)
        #continue
        

        #img = (orig_img[:,:,0]).copy()	
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

        cv2.imwrite("data/processed/colors/red/road"+str(road_sign_num)+"_r.png",output)
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
        #cv2.imwrite("data/processed/colors/red/road"+str(road_sign_num)+"_r.png",img)
        #continue

        """
        #img = cv2.Canny(r,0.6, 0.9)
        circles = cv2.HoughCircles(r.astype(np.uint8),cv2.HOUGH_GRADIENT, 2, 16, 0.9, 0.95)
        """
        
        img = cv2.GaussianBlur(img,(9,9),2)
        img = cv2.Canny(img,0.9, 0.95)

        #cv2.imwrite("data/processed/colors/red/road"+str(road_sign_num)+"_r.png",img)
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
        cv2.imwrite("data/processed/colors/red/road"+str(road_sign_num)+"_r.png",img)
        
# detect vertical lines in images -- signs in proximity
# cv.HoughLines

def merge_neighbor_colors(image: cv2.Mat):
    h, w=image.shape[:2]

    updated = np.zeros_like(image)
    for j in range(h):
        for i in range(w):
            neighbors = [[j-1,i], [j,i-1], [j,i+1], [j+1,i]]
            values = np.zeros_like(updated[:4,0])
            for ind in range(4):
                try:
                    neighbor_val = image[neighbors[ind]]
                    values[ind] = neighbor_val
                except Exception as e:
                    pass
            difs = np.abs(values - image[j,i])
            closest = values[np.argmin(difs)]
            updated[j,i] = np.rint((closest-image[j,i])/2.+image[j,i])
    
    return updated

def k_means_merge(img: cv2.Mat, K: int = 8):
    # Reshape the image to a 2D array of pixels
    Z = img.reshape((-1, 1))

    # Convert to float32
    Z = np.float32(Z)

        
    # Define termination criteria
    # (type, max_iter, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)

    # Define the number of clusters (generalized colors)
    # K = 8 # Reduce the image to 8 colors

    # Apply K-Means
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        
    # Convert back to uint8
    center = np.uint8(center)

    # Map the labels back to the color centers
    res = center[label.flatten()]

    # Reshape the result back to the original image dimensions
    quantized_img = res.reshape((img.shape))

    return quantized_img

if __name__ == "__main__":
    main()