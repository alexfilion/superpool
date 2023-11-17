import cv2
import numpy as np

from TableManualCalibration import TableManualCalibration
from BallsLocalisation import BallsLocalisation
from BallsDetection import BallsDetection

def CropImage(img, rectPts):
  pts = np.array(rectPts, np.int32)
  pts = pts.reshape((-1, 1, 2))
  mask = np.zeros_like(img)
  cv2.fillPoly(mask, [pts], (255, 255, 255))
  result = cv2.bitwise_and(img, mask)
  x, y, w, h = cv2.boundingRect(pts)
  cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0),2,1)
  cropped = result[y:y+h, x:x+w]
  return cropped



def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result



def main():


  TableManualCalibration = TableManualCalibration(videoSource=0)
  cornersPosition, angle = TableManualCalibration.Place4Corners()
  print(cornersPosition)

  cap = cv2.VideoCapture(0)
  
  while True:
    ret, frame = cap.read()

    if not ret:
      break

    ballsLocalisation = BallsLocalisation(frame=frame)
    positions = ballsLocalisation.FindPositions()




    cv2.imshow('Video', frame)


    key = cv2.waitKey(1)
    if key == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()






if __name__ == "__main__":
  main()
