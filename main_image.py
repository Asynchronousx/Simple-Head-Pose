import hpe
import cv2

# Creating an instance of the hpe and drawer model
model = hpe.SimplePose(model_type="svr")

# Train the model with the given dataset folder
#model.train("AFLW2000", save=True, split=0.1, ext="jpg")

# Load a pretrained model from the trained folder
model.load("best_model_svr_23_01_24_17")

# Load an image from the given path
image = cv2.imread("examples/faces_3.png")

# Flip the image horizontally for a later selfie-view display
# Also convert the color space from BGR to RGB
image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

# Get the result
poses, lms, bbox = model.predict(image)

# Draw the results on the image
cv2.imshow("Image", model.draw(image, poses, lms, bbox, draw_face=True, draw_person=False, draw_axis=True))
cv2.imwrite("example.png", model.draw(image, poses, lms, bbox, draw_face=True, draw_person=False, draw_axis=True))
cv2.waitKey(0)