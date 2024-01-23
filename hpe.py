import modelhub
import dataset
import drawer
import detector
import poseutils
import cv2

# Class that encapsulates the HPE model and its functionalities, 
# executing the entire pipeline to train, test and predict the pose parameters
class SimplePose:

    # Init the HPE model with the default objects
    def __init__(self, model_type="svr", mesh_type="mp", mesh_conf=0.25, mesh_iou=0.45, yolo_conf=0.25, yolo_iou=0.45):

        # We init an initial empty svr model 
        self.model = modelhub.load(
            model_name=model_type, 
            mesh_type=mesh_type, 
            mesh_conf=mesh_conf, 
            mesh_iou=mesh_iou
        )

        # Dataset object to load the dataset
        self.dataset = dataset.DataManager()

        # Detector object to detect people and faces
        self.detector = detector.Detector(yolo_conf, yolo_iou)

        # Drawer object to draw the results if needed
        self.drawer = drawer.Drawer()


    # Function to train the model 
    def train(self, dataset_folder, save=True, split=0.1, ext="jpg"):

        # We load the dataset with the given dataset folder path 
        # and the desired image format (default is jpg)
        self.dataset.assign_path(dataset_folder)
        x, y = self.dataset.load(ext=ext)

        # Split the dataset into train and test set (default test size is 0.2 and random state is 69)
        X_train, X_test, y_train, y_test = self.dataset.train_test_split(x, y, test_size=split)

        # Train and evaluate the model, save model is enabled by default
        self.model.train(X_train, y_train, save=save)
        self.model.eval(X_test, y_test)

    # Function to load a pretrained model
    def load(self, model_name):
            
            # We load the model from the given model name
            self.model.load(model_name)

    # Function to predict the pose parameters of a single image and the landmarks if specified
    def predict(self, image):

        # We check if the image is a string else we assume it is an image
        # and its already loaded and converted to rgb
        if isinstance(image, str):
            
            # If so, we need to load it a convert it to rgb while flipping it
            # to increase accuracy 
            image = cv2.imread(image)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Since the image could contain more than one face, we need to 
        # run a person to face detection step first to get the bounding boxes of the faces.
        # We then crop the faces from the image and pass them to the model to predict the pose.
        # Declaring a list of landmarks and poses, and a dictionary of bounding boxes
        landmarks = []
        poses = []
        bbox = {}

        # Detect people's faces in the image alongside the bounding box coordinates, either 
        # of their faces and body 
        faces, bbox = self.detector.detect(image)
        
        # We iterate over the faces list and predict the pose for each face
        for face in faces:

            # Since face is a dictionary with one key and one value, we need to extract them
            # to get the face id and the face image
            key = list(face.keys())[0]
            face = face[key]

            # Get the pose and landmarks from the model predictions
            pose, lms = self.model.predict(face, return_landmarks=True)

            # Init an empty scaled lms list
            scaled_lms = None

            # We check if the lms are not empty, otherwise we skip
            if lms is not None:

                # Unflatten the landmarks list from a list of float to a list of float tuples (x,y)
                lms = [(lms[i], lms[i+1]) for i in range(0, len(lms)-1, 2)]

                # Init an empty scaled lms list since if here, it means that we got some results
                scaled_lms = []

                # After we obtain the landmarks, since they're in the 0-1 range we need to 
                # scale them back to the original image size. We simply do this by multiplying
                # each coordinate by the image width and height respectively of the face image,
                # adding the offset of the face+person bounding box coordinates to get the  
                # correct coordinates referring to the original image.
                # For each landmark in lms
                for lm in lms:
                    
                    # Extract x and y 
                    x, y = lm

                    # Scale the coordinates for the size of the face image 
                    x = int(x * face.shape[1])
                    y = int(y * face.shape[0])

                    # Extract the face bounding box coordinates: we don't need person bbox coordinates
                    # to displace the offset since face bbox coordinates are already in the form of 
                    # face_coord_x1 + person_coord_x1, face_coord_y1 + person_coord_y1, etc.
                    x_offset, y_offset, _, _ = bbox[key]["face"]
                    
                    # Add the offset to the scaled coordinates
                    x = x + x_offset
                    y = y + y_offset

                    # Append the scaled coordinates to the scaled lms list
                    scaled_lms.append((x,y))           

            # Append those to the appropriate lists
            landmarks.append(scaled_lms)
            poses.append(pose)
    
        # Finally return the lists of landmarks and poses, alongside the bounding boxes
        return poses, landmarks, bbox

    # Function to draw the results on the image
    def draw(self, image, poses, lms, bbox, axis_size = 50, draw_face=True, draw_person=False, draw_lm=False, draw_axis=True):

        # Based on the type of draw, we draw the results on the image
        # using the drawer object
        if draw_person or draw_face:
            image = self.drawer.draw_bbox(image, bbox, draw_face=draw_face, draw_person=draw_person)
        
        if draw_lm:
            image = self.drawer.draw_landmarks(image, lms)

        if draw_axis:
            image = self.drawer.draw_axis(cv2.flip(image,1), poses, lms, bbox, axis_size)

        # Return the image with the results drawn
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        