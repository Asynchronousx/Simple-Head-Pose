import hpe
import cv2

# Creating an instance of the hpe model
model = hpe.SimplePose(
    model_type="svr", 
    mesh_type="mp", 
    mesh_conf=0.1, 
    mesh_iou=0.1,
    det_conf=0.12,
    det_iou=0.25
)

# Train the model with the given dataset folder
#model.train("AFLW2000", save=True, split=0.1, ext="jpg")

# Load a pretrained model from the trained folder
model.load("best_model_svr_18_01_24_19")

# Open the webcam or a videofile
cap = cv2.VideoCapture(0)

# Open a videofile
#cap = cv2.VideoCapture("examples/head_rotation.mp4")

# Init parameters for video writer  
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "example.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# While the webcam is open
while cap.isOpened():

    # Read the frame
    success, image = cap.read()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Get the result
    poses, lms, bbox = model.predict(image)

    # Draw the results on the image
    image = model.draw(image, poses, lms, bbox, draw_face=True, draw_person=False, draw_axis=True)
    cv2.imshow("Image", image)

    # write to video 
    out.write(image)

    # Exit if the user press ESC
    if cv2.waitKey(5) & 0xFF == 27:
        break

