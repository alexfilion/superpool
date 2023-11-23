import cv2
import numpy as np


class LocateBalls:
  def __init__(self, frame, corners):
    self.croppedFrame = self.CropImage(frame, corners)
    self.finalFrame, self.positions = self.ProcessImage(self.croppedFrame)


  def AdjustLuminosityExp(self, img, ordre=2):
    img = np.power(img.astype(np.uint32), ordre)
    img = img / (255**(ordre - 1))
    img = img.astype(np.uint8)
    return img


  def CropImage(self, frame, corners):
    pts = np.array(corners, np.int32) # Reorder the points in clockwise order
    pts = pts.reshape((-1, 1, 2))

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    result = cv2.bitwise_and(frame, mask)

    contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    croppedFrame = result[y:y+h, x:x+w]

    return croppedFrame


  def ProcessImage(self, frame):
    #To do or not
    
    frame2 = self.AdjustLuminosityExp(frame, 3)
    #display = np.hstack((frame, frame2))
    #cv2.imshow('image', display)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 5)
    
    #display = np.hstack((blurred, thresholded))
    #cv2.imshow('image', display)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    circles = cv2.HoughCircles(thresholded, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=5, param2=12, minRadius=7, maxRadius=15)

    positions = []
    if circles is not None:
      circles = np.uint16(np.around(circles))
      for circle in circles[0, :]:
        cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)
        cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 1)

        positions.append((circle[0], circle[1]))

    return frame, positions





videoCapture = cv2.VideoCapture(0)
videoFeed = True

while True:
  ret, frame = videoCapture.read()
  if not ret:
    break

  # Video Feed
  if videoFeed:
    locateBalls = LocateBalls(frame, [[36, 413], [19, 77], [671, 66], [665, 413]])
    print(locateBalls.positions)
    cv2.imshow("Final Frame", locateBalls.finalFrame)

  # Screen Shot
  else:
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
      locateBalls = LocateBalls(frame, [[36, 413], [19, 77], [671, 66], [665, 413]])
      print(locateBalls.positions)
      cv2.imshow("Final Frame", locateBalls.finalFrame)

  if cv2.waitKey(1) == 27:
    cv2.VideoCapture(0).release()
    cv2.destroyAllWindows()
    break
