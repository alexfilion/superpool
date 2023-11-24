import cv2
import numpy as np


class FindTableCorners:
  def __init__(self, videoSource=0):
    self.cap = cv2.VideoCapture(videoSource)

    cv2.namedWindow("image")
    cv2.createTrackbar("H Min", "image", 0, 180, lambda x: None)
    cv2.createTrackbar("H Max", "image", 180, 180, lambda x: None)
    cv2.createTrackbar("S Min", "image", 0, 255, lambda x: None)
    cv2.createTrackbar("S Max", "image", 174, 255, lambda x: None)
    cv2.createTrackbar("V Min", "image", 0, 255, lambda x: None)
    cv2.createTrackbar("V Max", "image", 255, 255, lambda x: None)

    self.corners = self.Run()


  def CalculateTableCorners(self, lines, imgWidth, imgHeight):
    corners = []
    for i in range(len(lines)):
      for j in range(i + 1, len(lines)):
        rho1, theta1 = lines[i][0]
        rho2, theta2 = lines[j][0]

        A = np.array([
          [np.cos(theta1), np.sin(theta1)],
          [np.cos(theta2), np.sin(theta2)]
        ])
        B = np.array([[rho1], [rho2]])

        if np.linalg.matrix_rank(A) >= 2: # Check if the matrix is not singular
          intersection = np.linalg.solve(A, B)
          #x, y = int(intersection[0]), int(intersection[1])
          x, y = map(int, intersection.flatten()[:2])

          if 0 <= x < imgWidth and 0 <= y < imgHeight:
            corners.append((x, y))
        
    cwCorners = self.ClockwiseCornersPosition(corners)
    return cwCorners


  ################################################################ To be tested ################################################################
  # Maybe not necessary
  def ClockwiseCornersPosition(self, corners):
    if len(corners) < 4:
        return corners
    
    #pts = np.array(corners, dtype=np.float32)
    #rect = cv2.minAreaRect(pts)
    #center, size, angle = rect
    #sortedCorners = sorted(pts, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
    # OR

    pts = np.array(corners, dtype=np.float32)
    centroid = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sortedIndices = np.argsort(angles)
    sortedCorners = pts[sortedIndices]

    return sortedCorners
  

  # If the lines are too close one to the other
  def filterCloseLines(lines, minDistance):
    filteredLines = []

    for line in lines:
      rho, theta = line[0]
      addLine = True

      # Check against existing lines in filteredLines
      for filteredLine in filteredLines:
        rhoFiltered, thetaFiltered = filteredLine[0]

        # If the lines are very close in parameter space, skip adding this line
        if abs(rho - rhoFiltered) < minDistance and abs(theta - thetaFiltered) < np.pi / 180 * minDistance:
          addLine = False
          break

      if addLine:
        filteredLines.append(line)

    return filteredLines



  def Run(self):
    corners = []
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break

      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      hMin = cv2.getTrackbarPos("H Min", "image")
      hMax = cv2.getTrackbarPos("H Max", "image")
      sMin = cv2.getTrackbarPos("S Min", "image")
      sMax = cv2.getTrackbarPos("S Max", "image")
      vMin = cv2.getTrackbarPos("V Min", "image")
      vMax = cv2.getTrackbarPos("V Max", "image")

      lowerHsv = np.array([hMin, sMin, vMin])
      upperHsv = np.array([hMax, sMax, vMax])

      maskHsv = cv2.inRange(hsv, lowerHsv, upperHsv)
      resultHsv = cv2.bitwise_and(frame, frame, mask=maskHsv)

      edges = cv2.Canny(maskHsv, 50, 150, apertureSize=3)
      lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)


      if lines is not None:
        corners = self.CalculateTableCorners(lines, edges.shape[1], edges.shape[0])

        for corner in corners:
          cv2.circle(frame, corner, 15, (0, 255, 0), 2)

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
          
          cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

          # To find the rule of the line
          #if (y2 - y1) != 0:
          #  m = (x2 - x1) / (y2 - y1)
          #  b = y1 - x1 * m
          #  print(f"y = {m}x + {b}")

      display = np.hstack((frame, resultHsv))
      cv2.imshow("image", display)

      if cv2.waitKey(1) == 13:
        self.cap.release()
        cv2.destroyAllWindows()
        return corners


findTableCorners = FindTableCorners()
print(findTableCorners.corners)
