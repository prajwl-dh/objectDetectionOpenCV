#Author: Prajwal Dhungana
#Last-Modified : 02/28/2023

# Import the necessary packages
import cv2
import numpy as np
import writeToDatabase as db

# Define range of colors for each cube
red_lower = np.array([155, 133, 84], dtype=np.uint8)
red_upper = np.array([179, 255, 255], dtype=np.uint8)
blue_lower = np.array([100, 50, 50], dtype=np.uint8)
blue_upper = np.array([124, 255, 255], dtype=np.uint8)
green_lower = np.array([70, 120, 64], dtype=np.uint8)
green_upper = np.array([86, 255, 255], dtype=np.uint8)
yellow_lower = np.array([25, 122, 94], dtype=np.uint8)
yellow_upper = np.array([42, 255, 255], dtype=np.uint8)
purple_lower = np.array([136, 72, 54], dtype=np.uint8)
purple_upper = np.array([150, 150, 191], dtype=np.uint8)
orange_lower = np.array([0, 140, 148], dtype=np.uint8)
orange_upper = np.array([180, 255, 255], dtype=np.uint8)

# Set up the capture object
cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Draw a circle at the center of the frame
    center_x = int(width / 2)
    center_y = int(height / 2)
    radius = 50
    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

    # Print the (x,y) coordinate of the center of the circle
    #print(f"Center of circle: x={center_x}, y={center_y}")

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)

    # Bitwise-AND mask and original image
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)
    purple = cv2.bitwise_and(frame, frame, mask=purple_mask)

    # Convert the masks to grayscale
    red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
    blue_gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    yellow_gray = cv2.cvtColor(yellow, cv2.COLOR_BGR2GRAY)
    purple_gray = cv2.cvtColor(purple, cv2.COLOR_BGR2GRAY)

    # Find contours in the grayscale masks
    red_contours, _ = cv2.findContours(red_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    purple_contours, _ = cv2.findContours(purple_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame for each color
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area < 2500 or area > 20000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            frame = cv2.drawContours(frame, [box], 0, (50, 0, 255), 2)
            cv2.putText(frame, 'Red color', (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            x, y, w, h = box.astype(int)
            # Print the x and y position of the red foam cube
            #x is the top left corner , y is the bottom right corner
            #print(f"Red foam cube position: x={x}, y={y}")
            
            # Calculate the centroid of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

                # Print the centroid coordinates
                #print("Red foam cube centroid position: x={}, y={}".format(cX, cY))
                
            if(cX > center_x + 50):
                print("Move Right")
            elif(cX < center_x - 50):
                print("Move Left")
            elif(cY > center_y + 50):
                print("Move Up")
            elif(cY < center_y - 50):
                print("Move Down")
            else:
                print("At Center")

    for cnt in blue_contours:
        area = cv2.contourArea(cnt)
        if area < 2500 or area > 20000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            frame = cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
            cv2.putText(frame, 'Blue color', (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            x, y, w, h = box.astype(int)
            # Print the x and y position of the red foam cube
            #x is the top left corner , y is the bottom right corner
            #print(f"Red foam cube position: x={x}, y={y}")
            
            # Calculate the centroid of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

                # Print the centroid coordinates
                #print("Red foam cube centroid position: x={}, y={}".format(cX, cY))
                
            if(cX > center_x + 50):
                print("Move Right")
            elif(cX < center_x - 50):
                print("Move Left")
            elif(cY > center_y + 50):
                print("Move Up")
            elif(cY < center_y - 50):
                print("Move Down")
            else:
                print("At Center")

    for cnt in green_contours:
        area = cv2.contourArea(cnt)
        if area < 2500 or area > 20000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            frame = cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv2.putText(frame, 'Green color', (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            x, y, w, h = box.astype(int)
            # Print the x and y position of the red foam cube
            #x is the top left corner , y is the bottom right corner
            #print(f"Red foam cube position: x={x}, y={y}")
            
            # Calculate the centroid of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

                # Print the centroid coordinates
                #print("Red foam cube centroid position: x={}, y={}".format(cX, cY))
                
            if(cX > center_x + 50):
                print("Move Right")
            elif(cX < center_x - 50):
                print("Move Left")
            elif(cY > center_y + 50):
                print("Move Up")
            elif(cY < center_y - 50):
                print("Move Down")
            else:
                print("At Center")

    for cnt in yellow_contours:
        area = cv2.contourArea(cnt)
        if area < 2500 or area > 20000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            frame = cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
            cv2.putText(frame, 'Yellow color', (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            x, y, w, h = box.astype(int)
            # Print the x and y position of the red foam cube
            #x is the top left corner , y is the bottom right corner
            #print(f"Red foam cube position: x={x}, y={y}")
            
            # Calculate the centroid of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

                # Print the centroid coordinates
                #print("Red foam cube centroid position: x={}, y={}".format(cX, cY))
                
            if(cX > center_x + 50):
                print("Move Right")
            elif(cX < center_x - 50):
                print("Move Left")
            elif(cY > center_y + 50):
                print("Move Up")
            elif(cY < center_y - 50):
                print("Move Down")
            else:
                print("At Center")

    for cnt in purple_contours:
        area = cv2.contourArea(cnt)
        if area < 2500 or area > 20000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            frame = cv2.drawContours(frame, [box], 0, (255, 0, 127), 2)
            cv2.putText(frame, 'Purple color', (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 127), 2)
            
            x, y, w, h = box.astype(int)
            # Print the x and y position of the red foam cube
            #x is the top left corner , y is the bottom right corner
            #print(f"Red foam cube position: x={x}, y={y}")
            
            # Calculate the centroid of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

                # Print the centroid coordinates
                #print("Red foam cube centroid position: x={}, y={}".format(cX, cY))
                
            if(cX > center_x + 50):
                print("Move Right")
            elif(cX < center_x - 50):
                print("Move Left")
            elif(cY > center_y + 50):
                print("Move Up")
            elif(cY < center_y - 50):
                print("Move Down")
            else:
                print("At Center")

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()