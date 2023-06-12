import cv2

# Load the image
img = cv2.imread("Coppercoin.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Detect keypoints and compute descriptors
kp, des = sift.detectAndCompute(gray, None)

# Draw keypoints on the image
img = cv2.drawKeypoints(img, kp, None)

# Display the image with keypoints
cv2.imshow("Image", img)
cv2.waitKey(0)