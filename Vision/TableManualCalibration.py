import cv2
import numpy as np


class TableManualCalibration:
  def __init__(self, videoSource=0):
    self.cap = cv2.VideoCapture(videoSource)
    self.dragging = False
    self.dragPointIdx = -1
    self.rightDragging = False
    self.rightClickPos = None

    self.radius = 25
    self.angle = 0
    self.rectPts = []

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', self.OnMouseEvent)


  def OnMouseEvent(self, event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
      self.dragPointIdx = self.GetDragPointIndex(x, y)
      if self.dragPointIdx != -1:
        self.dragging = True
        self.referencePoint = (x, y)
        self.rectBeforeDrag = self.rectPts.copy()

    elif event == cv2.EVENT_RBUTTONDOWN:
      self.rightClickPos = (x, y)
      self.rightDragging = True

    elif event == cv2.EVENT_LBUTTONUP:
      self.dragging = False
      self.dragPointIdx = -1

    elif event == cv2.EVENT_RBUTTONUP:
      self.rightDragging = False

    elif event == cv2.EVENT_MOUSEMOVE:
      if self.dragging:
        self.rectPts[self.dragPointIdx] = (x, y)

        dx = x - self.referencePoint[0]
        dy = y - self.referencePoint[1]

        if self.dragPointIdx == 0:  # Top-left corner
          self.rectPts[1] = (self.rectBeforeDrag[1][0], self.rectBeforeDrag[1][1] + dy)
          self.rectPts[3] = (self.rectBeforeDrag[3][0] + dx, self.rectBeforeDrag[3][1])
        elif self.dragPointIdx == 1:  # Top-right corner
          self.rectPts[0] = (self.rectBeforeDrag[0][0], self.rectBeforeDrag[0][1] + dy)
          self.rectPts[2] = (self.rectBeforeDrag[2][0] + dx, self.rectBeforeDrag[2][1])
        elif self.dragPointIdx == 2:  # Bottom-right corner
          self.rectPts[1] = (self.rectBeforeDrag[1][0] + dx, self.rectBeforeDrag[1][1])
          self.rectPts[3] = (self.rectBeforeDrag[3][0], self.rectBeforeDrag[3][1] + dy)
        elif self.dragPointIdx == 3:  # Bottom-left corner
          self.rectPts[0] = (self.rectBeforeDrag[0][0] + dx, self.rectBeforeDrag[0][1])
          self.rectPts[2] = (self.rectBeforeDrag[2][0], self.rectBeforeDrag[2][1] + dy)

      elif self.rightDragging:
        if self.rightClickPos is not None:
          self.RotateRectangle(x, y)


  def RotateRectangle(self, x, y):
    dx = x - self.rightClickPos[0]
    dy = y - self.rightClickPos[1]
    
    self.angle = np.arctan2(dy, dx)


  def GetDragPointIndex(self, x, y):
    for i, pt in enumerate(self.rectPts):
      if np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2) <= self.radius:
        return i
    return -1


  def Place4Corners(self):
    flagOnce = True
    while True:
      ret, frame = self.cap.read()

      if not ret:
        break

      if flagOnce:
        flagOnce = False
        height, width = frame.shape[:2]
        initialRectHeight = 200
        initialRectWidth = 400

        
        self.rectPts = [(width//2 - initialRectWidth//2, height//2 - initialRectHeight//2),
                        (width//2 + initialRectWidth//2, height//2 - initialRectHeight//2),
                        (width//2 + initialRectWidth//2, height//2 + initialRectHeight//2),
                        (width//2 - initialRectWidth//2, height//2 + initialRectHeight//2)]


      rect = cv2.minAreaRect(np.array(self.rectPts))
      box = np.int0(cv2.boxPoints(rect))
      
      M = cv2.getRotationMatrix2D(self.rectPts[0], np.degrees(self.angle), 1)
      box = cv2.transform(np.array([box]), M)[0]
      cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

      for corner in box:
        cv2.circle(frame, tuple(corner), self.radius, (0, 0, 255), 2)  # Draw a circle at each corner

      cv2.imshow('Video', frame)


      key = cv2.waitKey(1)
      if key == ord('q'):
        break
      elif key == 13:  # Check for "Enter" key press (ASCII code)
        break

    self.cap.release()
    cv2.destroyAllWindows()

    return box, self.angle
