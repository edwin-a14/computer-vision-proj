import cv2
import numpy as np
import os
import shutil
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
import matplotlib.colors as clr

def main():
    lower_bound_rgb, upper_bound_rgb,lower_bound_hsv, upper_bound_hsv = learn()
    #print(lower_bound_rgb,upper_bound_rgb)
    #lower_bound, upper_bound = ((77, 87, 108), (178, 189, 203))
    
    directory_path = "data/raw/kaggle_roadsign/images" 
    results_path = "data/processed/found_chips"
    
    os.makedirs(results_path, exist_ok=True)

    images = []
    try:
        entries = os.listdir(directory_path)
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path): 
                images.append(entry)
                path = os.path.join(results_path, os.path.splitext(entry)[0])
                os.makedirs(path,exist_ok=True)
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
    
    # try and detect circular red things to check first
    for road_sign_image in images:
        
        path = os.path.join(directory_path,road_sign_image)
        orig_img = cv2.imread(path)
        path = os.path.join(results_path,os.path.splitext(road_sign_image)[0])

        chip_counter = 0
        img = orig_img.copy()
        color_mask = cv2.inRange(img,lower_bound_rgb,upper_bound_rgb)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv_img, lower_bound_hsv, upper_bound_hsv)
        
        combined_mask = cv2.bitwise_or(color_mask,hsv_mask)
        reconstructed_img = cv2.bitwise_and(orig_img, orig_img, mask=combined_mask)
        local_path = chip_path(path, chip_counter, road_sign_image)

        
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_img[:,:,0] = edge_Sobel(hsv_img[:,:,0])
        img = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR)
        
        # Non-linear blurring preserving edges
        img = cv2.medianBlur(img, 5)    


        """
        # Blur colors
        img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        img = cv2.GaussianBlur(img, (75,75),5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_thresh = 50
        img[img<gray_thresh]=0
        """

        #img = cv2.Canny(img, 0.8, 0.9)
        


        color_image = np.zeros_like(orig_img)
        # Find Connected Regions
        img = cv2.Canny(img, 0.9, 0.95)

        regions = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
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
            if area > 500:
                # Draw the bounding box on the original image
                #cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cropped = orig_img[y:y+h,x:x+w]
                local_path = chip_path(path, chip_counter, road_sign_image)

                cv2.imwrite(local_path,cropped)
                chip_counter+=1

                #component_mask = (labels == i).astype(np.uint8)
                # Generate a random BGR color for the component
                # The randint function generates a tuple of three random integers [0, 255]
                #random_color = tuple(np.random.randint(0, 256, 3).tolist())

                # Use the component mask to set the pixels to the random color
                # This copies the random_color to all pixels where component_mask is 1
                #color_image[component_mask == 1] = random_color
                # Add a text label (optional)
                #cv2.putText(orig_img, "Component " + str(road_sign_num), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        
        
        remove_previous_chips(path, chip_counter, road_sign_image)
        continue
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
        """

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
        
def learn():
    directory = find_exact_contours()
    return collect_stats(directory)

def collect_stats(directory_path: str):
    """
        returns:
            lower, higher bgr bound
            lower, higher hsv bound
    """
    images = []
    try:
        entries = os.listdir(directory_path)
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path): 
                images.append(cv2.imread(full_path))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
    
    bounds = find_bounds(images)


    #print("Color Filter range: ",bgr_colors[len(bgr_colors)//9],'|',bgr_colors[-len(bgr_colors)//9])
    return bounds


    color_array = np.zeros((len(bgr_colors),3),dtype=np.uint8)
    index = 0
    for color in bgr_colors:
        value = clr.to_rgb(color)
        color_array[index]=np.array(list(value)[::-1])
        index+=1
    return (np.percentile(color_array,10),np.percentile(color_array, 90))

def find_bounds(images: list[cv2.Mat]):
    rgb_vals = []
    hsv_vals = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w, _ = image.shape
        for j in range(h):
            for i in range(w):
                r, g, b = tuple(image_rgb[j,i])
                hex_color = f"#{r:02X}{g:02X}{b:02X}"
                rgb_vals.append(hex_color)

                h, s, v = tuple(image_hsv[j,i])
                hex_color = f"#{h:02X}{s:02X}{v:02X}"
                hsv_vals.append(hex_color)


    #rgb_vals = list(rgb_vals)
    #hsv_vals = list(hsv_vals)
    rgb_vals.sort(key=lambda h: int(h[1:],16))
    hsv_vals.sort(key=lambda h: int(h[1:],16))
    
    lower_bound_rgb = (np.array(clr.to_rgb(rgb_vals[len(rgb_vals)//4]))*255).astype(np.uint8)
    upper_bound_rgb = (np.array(clr.to_rgb(rgb_vals[-len(rgb_vals)//4]))*255).astype(np.uint8)

    lower_bound_hsv = (np.array(clr.to_rgb(hsv_vals[len(hsv_vals)//4]))*255).astype(np.uint8)
    upper_bound_hsv = (np.array(clr.to_rgb(hsv_vals[-len(hsv_vals)//4]))*255).astype(np.uint8)

    return(lower_bound_rgb[::-1], upper_bound_rgb[::-1], lower_bound_hsv, upper_bound_hsv)

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

def dbscan_merge(img: cv2.Mat, epsilon:float = 50, minimum_samples: int = 10):
    # Reshape the image to a 2D array of pixels
    Z = img.reshape((-1, 1))

    # Convert to float32
    Z = np.float32(Z)

        
    # Apply DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=minimum_samples).fit(Z) # Adjust eps and min_samples as needed
    labels = db.labels_

    unique_labels = set(labels)
    colors = {}
    for k in unique_labels:
        if k != -1:  # Exclude noise points
            cluster_pixels = Z[labels == k]
            mean_color = np.mean(cluster_pixels, axis = 0)
            colors[k] = mean_color.astype(np.uint8)


    clustered_image = np.zeros_like(img, dtype=np.uint8)
    for k in unique_labels:
        if k != -1:
            clustered_image[labels == k] = colors[k]
        else:
            clustered_image[labels == k] = [0, 0, 0] # Noise as black
    
    clustered_image = clustered_image.reshape(img.shape)

    return clustered_image

def resize(img: cv2.Mat, scale_factor: float):
    return cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST_EXACT)

def edge_Sobel(img: cv2.Mat):
    # Edge detection
    edges1 = cv2.Sobel(img,cv2.CV_32F,2,0,ksize=3)
    edges2 = cv2.Sobel(img,cv2.CV_32F,0,2,ksize=3)

    magnitude = cv2.magnitude(edges1,edges2)
    normalized_edges = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return normalized_edges

def chip_path(directory: str, index: int, file_name: str):
    parts = os.path.splitext(file_name)
    return os.path.join(directory, parts[0]+'_'+ str(index)+parts[-1])

def remove_previous_chips(directory: str, index: int, file_name: str):
    while True:
        path = chip_path(directory, index, file_name)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                pass
            index+=1
        else:
            break

def find_exact_contours():
    directory_path = "data/processed/chips/train/stop" 
    results_path = "data/processed/known_chips/stop/contours"
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    
    os.makedirs(results_path, exist_ok=True)

    images = []
    try:
        entries = os.listdir(directory_path)
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path): 
                images.append(entry)
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
    
    for image in images:
        img = cv2.imread(os.path.join(directory_path,image))

        height, width, _ = img.shape
        center = (width // 2, height // 2)
        
        coefficient = 0.9
        radius = round(min(height,width)*coefficient) // 2
        
        covered_image = img.copy()
        covered_image=replace_color(covered_image, (0,0,0),(50,50,50),(255,255,0))
        
        mask_size = (round(width*coefficient / 2)*2, round(height*coefficient / 2)*2)

        mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,mask_size).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (radius*2-1,radius*2-1),5)
        
        # To see how close the blurred mask is to an octagon
        #cv2.imwrite(os.path.join(results_path,image), cv2.normalize(mask,cv2.CV_32F,0,255,cv2.NORM_MINMAX,cv2.CV_8U))
        #continue

        area_to_blur = covered_image[center[1]-mask_size[1]//2:center[1]+mask_size[1]//2, center[0]-mask_size[0]//2:center[0]+mask_size[0]//2]
        #blurred_area = cv2.GaussianBlur(area_to_blur, (radius*2-1,radius*2-1),15)
        for j in range(mask_size[1]):
            for i in range(mask_size[0]):
                #area_to_blur[j,i] = ((blurred_area[j,i].astype(np.float32)-area_to_blur[j,i].astype(np.float32))*mask[j,i]+area_to_blur[j,i].astype(np.float32)).astype(np.uint8)
                area_to_blur[j,i] = ((1-mask[j,i])*area_to_blur[j,i].astype(np.float32)).astype(np.uint8)

        hls_img = cv2.cvtColor(covered_image, cv2.COLOR_BGR2HLS)
        hls_img[:,:,2]=cv2.medianBlur(hls_img[:,:,2],3)
        hls_img[:,:,1]=cv2.medianBlur(hls_img[:,:,1],5)
        hls_img = k_means_merge(hls_img,3)
        reduced_img = cv2.cvtColor(hls_img, cv2.COLOR_HLS2BGR)
        gray_image = cv2.cvtColor(reduced_img, cv2.COLOR_BGR2GRAY)

        thresh = np.zeros((height,width)).astype(np.uint8)
        thresh[gray_image<50] = 255

        contour_mask = np.zeros_like(gray_image) 
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(contour_mask, [largest_contour], 0, 255, -1)

        new_img = np.zeros_like(img)
        new_img[contour_mask>0]=img[contour_mask>0]

        cv2.imwrite(os.path.join(results_path,image), new_img)
    return results_path

def replace_color(img: cv2.Mat, lower_bound: tuple, upper_bound: tuple, to_color: tuple):
   
    mask1 = cv2.inRange(img, np.array(lower_bound), np.array(upper_bound))
    mask2 = cv2.bitwise_not(mask1)

    to_color = np.array(to_color, dtype=np.uint8)
    color_image = np.full_like(img, to_color)

    color_replacement = cv2.bitwise_and(color_image, color_image, mask=mask1)
    original_not_color = cv2.bitwise_and(img, img, mask=mask2)

    return cv2.add(original_not_color, color_replacement)

if __name__ == "__main__": 
    main()