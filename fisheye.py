import cv2
import numpy as np
import os
import discorpy.post.postprocessing as post

# xcenter = 782.5732073914905
# ycenter = 539.8631224869504
# factors = [1.0154161165196605, -0.00012324049796004737, 1.1062875290233011e-07, -1.141523930243955e-09, 1.0148972581951193e-12]

xcenter = 785.3054074949058
ycenter = 530.4032340121979
factors = [ 1.00851460e+00, -1.63425112e-05, -3.77548357e-07, -7.29162454e-11, 1.96865934e-13]

# xcenter = 756.8088684316172
# ycenter = 526.47363436134
# factors = [1.02002335e+00,  4.41166940e-05, -7.61327566e-07,  6.07328534e-10, -2.00542551e-13]

# xcenter = 782.5732073914905
# ycenter = 539.8631224869504
# factors = [1.0154161165196605, -0.00012324049796004737, 1.1062875290233011e-07, -1.141523930243955e-09, 1.0148972581951193e-12]

def calibration():
    # Number of inner corners in the chessboard pattern
    num_corners_width = 8  # Number of inner corners per chessboard row
    num_corners_height = 5  # Number of inner corners per chessboard column

    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    # This creates a grid of points for a 9x6 chessboard
    objp = np.zeros((num_corners_width * num_corners_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners_width, 0:num_corners_height].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    # List of calibration images
    images_path = '/Users/jibly/Documents/labHandling/lab_handling_system/chess'
    images = os.listdir(images_path)  # You need to provide the paths to your calibration images here
    # print(images)
    dimension = None
    for image in images:

        if not image.lower().endswith('.jpg'): 
            continue

        img_path = os.path.join(images_path, image)

        print(f'File: {img_path}')

        # Read the image from file
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Convert the image to grayscale as the corner detection works better on grayscale images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if dimension is None:
            dimension = gray.shape[::-1]

        # Find the chessboard corners
        # The function returns a boolean indicating if the pattern was found and the coordinates of the corners
        ret, corners = cv2.findChessboardCorners(gray, (num_corners_width, num_corners_height), None)

        # If the corners are found in the image
        if ret == True:
            print('yay')
            # Add the object points (same for all images since the chessboard pattern is the same)
            objpoints.append(objp)

            # Refine the corner locations (optional but recommended to improve accuracy)
            # This step uses the grayscale image, initial corner locations, and a window size for the search
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Add the refined image points
            imgpoints.append(corners2)

            # Optional: Draw and display the corners to check if they are detected correctly
            img = cv2.drawChessboardCorners(img, (num_corners_width, num_corners_height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)  # Display each image for 500 ms

    cv2.destroyAllWindows()

    # add new dimension to circumvent some weird dimension error when calling calibrate
    objpoints = np.expand_dims(np.asarray(objpoints), -2)

    # Calibration: Estimate the camera intrinsic parameters and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, dimension, None, None
    )

    # Print out the calibration results
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # save these values to use for actual callibration

# undistort image
# mtx - intrinstic camera matrix
# dist - distorion coefficients
def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    # newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    #     mtx, dist, (w,h), np.eye(3), balance=1)
    # undistorted_img = cv2.fisheye.undistortImage(img, mtx, dist,None, newcameramtx)

    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (w,h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img

def unwarp(img):
    img_corrected = np.copy(img)
    for i in range(img.shape[-1]):
        img_corrected[:, :, i] = post.unwarp_image_backward(img[:, :, i], xcenter,
                                                            ycenter, factors)
    return img_corrected