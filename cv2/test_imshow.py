import cv2

fname = "../data/car.jpg"
color = cv2.imread(fname, cv2.IMREAD_COLOR)
if color is not None:
    print("imread success")
cv2.imshow("Color image", color)
cv2.waitKey(0)
cv2.destroyAllWindows()
