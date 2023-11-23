import cv2
import numpy as np

from Utils.Functions import ResizeImage





class IsolateTable():
   def __init__():
      pass
   


def FindIntersection(a1, b1, a2, b2):
  x = (b1 - b2) / (a2 - a1)
  y = a1 * x + b1
  return (x, y)

def nothing(x):
  pass

def ResizeImage(image, width=700):
  return cv2.resize(image, (width, int(image.shape[0] * (width / image.shape[1]))))


def find_rectangle_corners(lines, img_width, img_height):
  corners = []
  for i in range(len(lines)):
      for j in range(i + 1, len(lines)):
          rho1, theta1 = lines[i][0]
          rho2, theta2 = lines[j][0]

          A = np.array([
              [np.cos(theta1), np.sin(theta1)],
              [np.cos(theta2), np.sin(theta2)]
          ])

          b = np.array([[rho1], [rho2]])

          intersection = np.linalg.solve(A, b)
          x, y = int(intersection[0]), int(intersection[1])

          if 0 <= x < img_width and 0 <= y < img_height:
              corners.append((x, y))

  return corners



cv2.namedWindow('image')
cv2.createTrackbar('H Min', 'image', 0, 180, lambda x: None)
cv2.createTrackbar('H Max', 'image', 180, 180, lambda x: None)
cv2.createTrackbar('S Min', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('S Max', 'image', 174, 255, lambda x: None)
cv2.createTrackbar('V Min', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('V Max', 'image', 255, 255, lambda x: None)

img = cv2.imread("Table8Balls_H.png")
img = ResizeImage(img)
imgCopy = ResizeImage(img)

while True:
  imgCopy = img.copy()
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  h_min = cv2.getTrackbarPos('H Min', 'image')
  h_max = cv2.getTrackbarPos('H Max', 'image')
  s_min = cv2.getTrackbarPos('S Min', 'image')
  s_max = cv2.getTrackbarPos('S Max', 'image')
  v_min = cv2.getTrackbarPos('V Min', 'image')
  v_max = cv2.getTrackbarPos('V Max', 'image')

  lower_hsv = np.array([h_min, s_min, v_min])
  upper_hsv = np.array([h_max, s_max, v_max])

  mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
  result_hsv = cv2.bitwise_and(img, img, mask=mask_hsv)

  edges = cv2.Canny(mask_hsv, 50, 150, apertureSize=3)
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

  if lines is not None:
      corners = find_rectangle_corners(lines, edges.shape[1], edges.shape[0])
      for corner in corners:
          cv2.circle(imgCopy, corner, 15, (0, 255, 0), 2)
      for line in lines:
          rho, theta = line[0]
          a = np.cos(theta)
          b = np.sin(theta)
          x0 = a * rho
          y0 = b * rho
          x1 = int(x0 + 1000 * (-b))
          y1 = int(y0 + 1000 * (a))
          x2 = int(x0 - 1000 * (-b))
          y2 = int(y0 - 1000 * (a))

          if (y2 - y1) != 0:
              m = (x2 - x1) / (y2 - y1)
              b = y1 - x1 * m
              print(f"y = {m}x + {b}")
              cv2.line(imgCopy, (x1, y1), (x2, y2), (0, 0, 255), 2)

  display = np.hstack((imgCopy, result_hsv))
  cv2.imshow('image', display)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      print(corners)
      break

cv2.destroyAllWindows()









