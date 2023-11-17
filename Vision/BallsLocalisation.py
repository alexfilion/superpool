import cv2
import numpy as np

def nothing(x):
  pass

def resize_image(image, width=700):
  return cv2.resize(image, (width, int(image.shape[0] * (width / image.shape[1]))))

# Création d'une fenêtre OpenCV
cv2.namedWindow('image')

# Création des trackbars pour ajuster les valeurs HSV
cv2.createTrackbar('H Min', 'image', 0, 180, nothing)
cv2.createTrackbar('H Max', 'image', 180, 180, nothing)
cv2.createTrackbar('S Min', 'image', 0, 255, nothing)
cv2.createTrackbar('S Max', 'image', 174, 255, nothing)
cv2.createTrackbar('V Min', 'image', 0, 255, nothing)
cv2.createTrackbar('V Max', 'image', 255, 255, nothing)


# Lecture d'une image de test (remplacez ceci par votre image de billard)
img = cv2.imread("Table8Balls_H.png")
img = resize_image(img)
imgCopy = resize_image(img)


while True:
  imgCopy = img.copy()
  # Convertir l'image en HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Obtenir les valeurs actuelles des trackbars pour HSV
  h_min = cv2.getTrackbarPos('H Min', 'image')
  h_max = cv2.getTrackbarPos('H Max', 'image')
  s_min = cv2.getTrackbarPos('S Min', 'image')
  s_max = cv2.getTrackbarPos('S Max', 'image')
  v_min = cv2.getTrackbarPos('V Min', 'image')
  v_max = cv2.getTrackbarPos('V Max', 'image')

  # Définir la plage HSV à filtrer
  lower_hsv = np.array([h_min, s_min, v_min])
  upper_hsv = np.array([h_max, s_max, v_max])

  # Appliquer le masque HSV
  mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
  result_hsv = cv2.bitwise_and(img, img, mask=mask_hsv)




  # Afficher l'image HSV et l'image RGB côte à côte
  display = np.hstack((imgCopy, result_hsv))
  cv2.imshow('image', display)

  # Sortir de la boucle si 'q' est pressé
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()