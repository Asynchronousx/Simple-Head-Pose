# File in which we store some useful functions and data structure
# to be externally accessed by other modules
import mediapipe as mp
import numpy as np

# Dictionary containing the list of important landmarks 
# and their corresponding index in the shape vector
class LMinfo:

    def __init__(self):
        # We assign to the class the following attributes
        self.NOSE = 1
        self.FOREHEAD = 10
        self.LEFT_EYE = 33
        self.MOUTH_LEFT = 61
        self.CHIN = 199
        self.RIGHT_EYE = 263
        self.MOUTH_RIGHT = 291

    def get(self):
        # We return the list of landmarks
        return [self.NOSE, 
                self.FOREHEAD, 
                self.LEFT_EYE, 
                self.MOUTH_LEFT, 
                self.CHIN, 
                self.RIGHT_EYE, 
                self.MOUTH_RIGHT]                   

# Class scaler that will take care of scaling and normalizing landmarks
class Scaler:

    # Function to scale the landmakrs to a normalized range
    def scale(self, landmarks):

        # Since we extract the landmarks from the image, we need to scale them
        # in a common range so that the model can learn from them.
        # We do this in two way: 
        # The first one, is to scale the landmarks around the center of the face,
        # that is the point 1 (the nose tip). This is done by simply subtracting
        # to each landmark the coordinates of the nose point thats located at the 
        # start of the landmarks list.
        nose_point = landmarks[0]
        landmarks = [lm - nose_point for lm in landmarks]

        # We then scale the landmarks by dividing them by the distance between the
        # forehead point and the chin point. This is done by dividing the landmarks
        # by the distance between the forehead point and the chin point; in this way 
        # we make the landmarks independent from the scale of the image.
        # We first calculate the distance between the forehead point and the chin point
        forehead_point = landmarks[1]
        chin_point = landmarks[4]

        # We calculate the distance between the two points
        reference_length = np.linalg.norm(forehead_point - chin_point)

        # We then divide the landmarks by the reference length
        landmarks = landmarks / reference_length

        # We return the scaled landmarks
        return landmarks


# Class that wrap the mediapipe mesher object and performs the landmark extraction
class MPMesher:

    # Constructor
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        # We init a landmarks_info object
        self.landmarks_info = LMinfo()

        # We init the FaceMesh object from mediapipe
        self.mesher = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence, 
            min_tracking_confidence=min_tracking_confidence)

    # Function to get the landmarks from a given image
    def process(self, image):

        # We process the image with the face mesh model
        results = self.mesher.process(image)

        # We iterate through the faces detected in the image (should be one anyway since
        # we're using face crops from image detection, but we do iterate anyway since results
        # is a list, even if it contains only one element)
        # If some faces has been detected
        if results.multi_face_landmarks:

            # Iterate through faces (should be one anyway)
            for face_landmarks in results.multi_face_landmarks:

                # We create a list to store the landmarks
                landmarks = []

                # We iterate through the landmarks detected in the face
                for idx, lm in enumerate(face_landmarks.landmark):

                    # Considering the index (each index represent a specific point) we take the most 
                    # important face landmark (if we'd like to increase the number of landmarks, we can
                    # add more indexes to the list in utils)
                    if idx in self.landmarks_info.get():
                        
                        # We append the x and y coordinates of the landmark to the list
                        landmarks.append((lm.x, lm.y))

                # Convert the tuple format to a flat list format
                landmarks_flattened = [item for tup in landmarks for item in tup]

                # We return the landmarks
                return landmarks_flattened
            
            else: 
                
                # If no result, we return None
                return None


##### ADD OTHER MESHERS HERE 
# class mymesher:
#     def __init__(self):
#         self.landmarks_info = LMinfo()
#         self.mesher = mymesher()
#     def process(self, image):
#         ...
#         landmark = self.mesher.process(image)
#         if landmark in self.landmarks_info.get():
#           process landmark in some ways
#         ...
#         return landmark
#####


# Class that will wrap the mesher object from various methods used.
# The return of this class should always be a flattened 1D list of x,y landmark coordinates.
class Mesher:

    # Constructor for the mesher object (default is MEDIAPIPE)
    def __init__(self, mesher_type="mp", min_detection_confidence=0.5, min_tracking_confidence=0.5):

        # We init the FaceMesh object selecting the type of mesher
        self.mesher = self.__initmesher(mesher_type, min_detection_confidence, min_tracking_confidence)

    # Function that takes care of initializing different type of meshers.
    # This is done because we might want to use different meshers for different results.
    def __initmesher(self, mesher_type, min_detection_confidence, min_tracking_confidence):

        # If the mesher type is mediapipe
        if mesher_type == "mp":
                
                # We init the FaceMesh object from mediapipe
                return MPMesher(min_detection_confidence, min_tracking_confidence)

        # ELIF: Other meshers can be added here if implemented in the poseutils library
        # elif mesher_type == "other_mesher":
        #    return mesher_model_that_can_do_process_function(parameters if needed)

    # Function to get the landmarks from a given image
    def process(self, image):

        # Since this class is a wrapper for different mesher, we simply call the process
        # function of the mesher object. Note that, the return of this function should always
        # be a flattened 1D list of x,y landmark coordinates.
        return self.mesher.process(image)


