import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
from deskew import determine_skew
from matplotlib import pyplot as plt

imgsrc = "E:/VIT/TARP/AHE Image.jpg"
img = cv2.imread(imgsrc)
def findScore(img, angle):
 
    data = inter.rotate(img, angle, reshape = False, order = 0)
    hist = np.sum(data, axis = 1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skewCorrect(img):

    #Crops down the skewImg to determine the skew angle
    img = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)

    delta = 1
    limit = 45
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = findScore(img, angle)
        scores.append(score)
    bestScore = max(scores)
    bestAngle = angles[scores.index(bestScore)]
    bestAngle = determine_skew(img)
    rotated = inter.rotate(img, bestAngle, reshape = False, order = 0)
    print("[INFO] angle: {:.3f}".format(bestAngle))
    cv2.imshow("Original", img)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)
    
    #Return img
    return rotated
rot = skewCorrect(img)
plt.imshow(rot)
plt.title('Deskewed Image')
plt.show()
cv2.imwrite("Deskewed Image.jpg", rot)
cv2.waitKey()