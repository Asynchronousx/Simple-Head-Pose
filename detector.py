import torch
import cv2
import numpy as np

# Class that takes care of recognizing people and their faces in an image 
class Detector:

    # Function to initialize the class: default model is yolov5large
    # to perform better on small objects (like faces in video surveilance)
    def __init__(self, confidence=0.25, iou=0.45):
            
            # Load the YOLOv5 generic model for person detection
            self.pmodel = torch.hub.load('ultralytics/yolov5', 'yolov5l')

            # Load the YOLOv5 model for face detection trained on WIDER dataset
            self.fmodel = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5_faces')

            # Setting the confidence and iou threshold for the models
            self.pmodel.conf = confidence
            self.pmodel.iou = iou
            self.fmodel.conf = confidence
            self.fmodel.iou = iou

    # Function to set the confidence and iou threshold for the person model 
    def set_person_threshold(self, confidence, iou):
        self.pmodel.conf = confidence
        self.pmodel.iou = iou

    # Function to set the confidence and iou threshold for the face model
    def set_face_threshold(self, confidence, iou):
        self.fmodel.conf = confidence
        self.fmodel.iou = iou
    
    # Function to detect people and faces in an image. This function returns a list of
    # faces images and a list of bounding box coordinates for the detected people and faces
    def detect(self, image_rgb):

        # Run inference on the image
        results = self.pmodel(image_rgb)

        # Get the class labels and bounding box coordinates for all detected objects
        class_labels = results.pandas().xyxy[0]['name'].tolist()
        bbox_coordinates = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax']].values

        # Faces image list of detected people
        faces_imgs = []

        # Bounding box coordinates for the detected person and face
        bb_dict = {}

        # Dictionary id based on how many person are found in the image
        dict_id = 0

        # Iterate over all detected objects and draw the bounding box around only the person
        # For the range of possible classes
        for i in range(len(class_labels)):

            # If the class is person
            if class_labels[i] == 'person':

                # Create an entry in the dictionary for the person
                bb_dict[dict_id] = {}

                # Extract the bounding box coordinates from the ith entry
                px1, py1, px2, py2 = bbox_coordinates[i]

                # Append the bounding box coordinates to the dictionary
                bb_dict[dict_id]["person"] = [px1, py1, px2, py2]

                # Extract the subimage using array slicing and bb coordinates
                subimage = image_rgb[int(py1):int(py2), int(px1):int(px2)]

                # Run inference on the subimage
                fresults = self.fmodel(subimage)

                # Get the bounding box coordinates for all detected faces
                face_bbox_coordinates = fresults.xyxy[0][:, :4].detach().numpy()

                # Iterate over the detected face (if any) of the extracted subimage 
                # of the person in the original image
                for bbox in face_bbox_coordinates:
                    
                    # Extract the bounding box coordinates from the ith entry
                    # (and basically the only one since its a subimage of a person)
                    fx1, fy1, fx2, fy2 = bbox

                    # Append the bounding box coordinates to the dictionary: 
                    # Before doing so, we need to add the person bounding box coordinates
                    # to the face bounding box coordinates to get the correct coordinates
                    # referring to the original image, since the coordinates are relative
                    # to the subimage and not the original one.
                    bb_dict[dict_id]["face"] = [fx1+px1, fy1+py1, fx2+px1, fy2+py1]

                    # Extract the face image using array slicing and bb coordinates            
                    faceimage = subimage[int(fy1):int(fy2), int(fx1):int(fx2)]

                    # Append the face image to the list of faces including its ids
                    # for future references
                    faces_imgs.append({dict_id: faceimage})

                # Incrase the person counter
                dict_id += 1

        # Return the list of faces images and the dictionary of bounding box coordinates
        return faces_imgs, bb_dict


### CLASS TESTER ###
# uncomment this and run the facedet.py script to see some results #
'''
# Load the image, cvt to rgb and detect faces using the model
image = cv2.imread("image_path.ext")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
facedet = Detector()
faces, bb_dict = facedet.detect(image_rgb)

# Print the dictionary of bounding box coordinates and display the faces
print(bb_dict)
for face in faces:
    cv2.imshow('Face', face)
    cv2.waitKey(0)

# Draw the bounding box around the person and face.
# Iterate through the dictionary
for key in bb_dict.keys():

    print(bb_dict[key])

    # Extract the person bounding box coordinates
    px1, py1, px2, py2 = bb_dict[key]["person"]

    # Draw the bounding box around the person
    cv2.rectangle(image, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), 2)

    # Extract the face bounding box coordinates if present
    # (If a person has been detected not necessarily means that a face has been detected
    # too, so we need to check if the face key is present in the dictionary)
    if "face" in bb_dict[key]:    
        fx1, fy1, fx2, fy2 = bb_dict[key]["face"]

        # Draw the bounding box around the face
        cv2.rectangle(image, (int(fx1), int(fy1)), (int(fx2), int(fy2)), (0, 255, 0), 2)
    
# Display the image with bounding box around the person
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
### END CLASS TESTER ###

