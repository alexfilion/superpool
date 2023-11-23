import cv2
import numpy as np

from TableManualCalibration import TableManualCalibration






def main():


  TableManualCalibration = TableManualCalibration(videoSource=0)
  cornersPosition, angle = TableManualCalibration.Place4Corners()
  print(cornersPosition)

  cap = cv2.VideoCapture(0)
  
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    cv2.imshow('Video', frame)





    if cv2.waitKey(1) == 27:
      break

  cap.release()
  cv2.destroyAllWindows()





if __name__ == "__main__":
  main()
