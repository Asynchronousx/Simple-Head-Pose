import hpe
import drawer
import cv2

# Creating an instance of the hpe and model with the default model type (svr)
# Other model types are "xgboost" and "svr" and can be specified in the constructor
model = hpe.SimplePose()

# We also create a drawer to draw the results
drawer = drawer.Drawer()

# Load or train the model: 
# If we train the model, we need to specify the pat in which the dataset is placed.
# Note that the dataset must be in the correct format (see the dataset.py and model.py 
# file for more info about the structure and the flags) 
#model.train("AFLW2000", save=True, split=0.1, ext="jpg")

# Alternatively, we can load a pretrained model by passing the name of the model itself
# (without extension and folder, managed by default by the model class)
model.load("best_model_svr_18_01_24_19")

# Test the model with a given image 
image_path = "your_image_path"
poses, lms, bbox = model.predict(image_path)

# Print results
#print("Poses: {}".format(poses))
print("Landmarks: {}".format(lms))
#print("Bounding boxes: {}".format(bbox))

# Draw the bounding boxes around the faces and persons 
cv2.imshow("Image", drawer.draw_bbox(image_path, bbox))
cv2.waitKey(0)

# Draw the landmarks on the image
cv2.imshow("Image", drawer.draw_landmarks(image_path, lms))
cv2.waitKey(0)

# Draw the axis based on the pose parameters
cv2.imshow("Image", drawer.draw_axis(image_path, poses, lms, bbox, axis_size=50))
