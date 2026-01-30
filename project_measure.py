import numpy as np
import json
import cv2

# 1. Load Calibration Data
with open("camera_params.json", "r") as f:
    data = json.load(f)
    mtx = np.array(data["camera_matrix"])

# Extract focal length and center
fx = mtx[0, 0]
fy = mtx[1, 1]

# 2. Setup Experiment Constants
Z = 2750.0  # measured distance 2.75m
img_path = 'test_images/IMG_9675.JPG'

# 3. Mouse Callback Setup
points = []
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        # Draw a small circle where you clicked
        cv2.circle(temp_img, (x, y), 10, (0, 0, 255), -1)
        cv2.imshow("Click Top-Left then Bottom-Right", temp_img)

# 4. Image Loading and Selection
image = cv2.imread(img_path)
if image is None:
    print("Error: Could not find image at", img_path)
    exit()

temp_img = image.copy()
cv2.namedWindow("Click Top-Left then Bottom-Right")
cv2.setMouseCallback("Click Top-Left then Bottom-Right", select_points)

print("INSTRUCTIONS:")
print("1. Click the Top-Left corner of your object.")
print("2. Click the Bottom-Right corner of your object.")
print("3. Press any key once you have clicked both.")

while len(points) < 2:
    cv2.imshow("Click Top-Left then Bottom-Right", temp_img)
    if cv2.waitKey(1) & 0xFF == 27: # Press ESC to cancel
        break

cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. Math Calculations
if len(points) == 2:
    u1, v1 = points[0]
    u2, v2 = points[1]
    
    # Calculate pixel differences
    width_p = abs(u2 - u1)
    height_p = abs(v2 - v1)
    
    # Perspective Projection Formula: Real_Dim = (Pixels * Distance) / Focal_Length
    real_width = (width_p * Z) / fx
    real_height = (height_p * Z) / fy
    
    print("-" * 40)
    print(f"Calculated Width: {real_width:.2f} mm")
    print(f"Calculated Height: {real_height:.2f} mm")
    print("-" * 40)

    # 6. Visualization for Video
    cv2.rectangle(image, points[0], points[1], (0, 255, 0), 8)
    label = f"W: {real_width:.1f}mm, H: {real_height:.1f}mm"
    cv2.putText(image, label, (u1, v1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    
    # Show final result
    result_small = cv2.resize(image, (800, 600))
    cv2.imshow("Final Result", result_small)
    cv2.imwrite("validation_result.jpg", image)
    cv2.waitKey(0)
else:
    print("Selection cancelled.")