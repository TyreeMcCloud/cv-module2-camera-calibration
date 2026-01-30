import numpy as np
import cv2
import glob
import json
import os

# 1. Configuration
CHECKERBOARD = (6, 9)
SQUARE_SIZE = 25.0  # mm
RESIZE_FACTOR = 0.25 # Shrink to 25% for speed

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

image_folder = 'calibration_images'
images = glob.glob(os.path.join(image_folder, '*.jpg')) + \
         glob.glob(os.path.join(image_folder, '*.JPG'))

if not images:
    print(f"Error: No images found. Ensure your JPGs are in '{image_folder}'")
    exit()

print(f"Found {len(images)} images. Processing...")

for fname in images:
    img = cv2.imread(fname)
    if img is None: continue

    # Resize for speed
    small_img = cv2.resize(img, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

    # Use FAST_CHECK to skip bad images quickly
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, cv2.CALIB_CB_EXHAUSTIVE)

    if ret:
        objpoints.append(objp)
        # Refine corners
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        #Scale corners back up to the original image size
        imgpoints.append(corners2 / RESIZE_FACTOR)

        # Visual feedback
        cv2.drawChessboardCorners(small_img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Detecting...', small_img)
        cv2.waitKey(50)
    else:
        print(f"Skipped: Could not find corners in {os.path.basename(fname)}")

cv2.destroyAllWindows()

if len(objpoints) > 0:
    # Use the ORIGINAL image resolution for the final calibration
    original_shape = cv2.imread(images[0]).shape[:2][::-1]

    print(f"\nCalibrating based on {len(objpoints)} successful images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, original_shape, None, None
    )

    calibration_data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist(),
        "resolution": original_shape
    }

    with open("camera_params.json", "w") as f:
        json.dump(calibration_data, f)

    print(f"Success! Focal Length (fx): {mtx[0,0]:.2f}")
    print("Results saved to 'camera_params.json'")
else:
    print("Error: No corners detected in any images.")