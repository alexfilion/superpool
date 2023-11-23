import cv2
import numpy as np




def show_image(image, title='Image', width=700):
  # Display the image
  cv2.imshow(title, cv2.resize(image, (width, int(image.shape[0] * (width / image.shape[1])))))
  cv2.waitKey(0)
  cv2.destroyAllWindows()



image = cv2.imread('Table8Balls_H.png')  # Replace 'pool_table.jpg' with your image file

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(gray, (3, 3), 2, 2)

show_image(gray_blurred)

# Apply Hough Circle Transform
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=20, param2=30, minRadius=10, maxRadius=30)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    for circle in circles[0, :]:
        # Draw the outer circle
        cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)

# Display the detected circles
show_image(image)
