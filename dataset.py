from sklearn.model_selection import train_test_split
import pandas as pd
import mediapipe as mp
import poseutils
import scipy.io
import tqdm
import json
import cv2
import os

# Create a class for the data manager. This class has the simple task to manage,
# extract and handle the data extracted from the AFLW2000 dataset (and really, 
# every dataset with that given image-mat format).
class DataManager:

    def __init__(self, datapath=""):

        # Note that, as stated before, the data folder could be every dataset that we want 
        # to use, as long the data is in the correct format:
        # - Images for the facial landmark 
        # - Mat files from which the target values are extracted (yaw, pith, roll).
        # So something like image1.jpg and image1.mat for example.
        self.datapath = datapath

        # Creating a landmarks_info object to access the landmarks list
        self.landmarks_info = poseutils.LMinfo()

        # Creating a scaler object to scale and normalize results if needed
        self.scaler = poseutils.Scaler()

        # Creating a mediapipe face mesh object to extract the landmarks
        self.mesher = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.25, 
            min_tracking_confidence=0.45)


    # Function to set the data folder path
    def assign_path(self, datapath):
        self.datapath = datapath

    # Function that receives a list of landmarks from the mediapipe library and returns
    # a list of flattened 1d list of important landmarks
    def get_landmarks(self, results):

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
                        
                        # We transform the lm coordinates from normalized to pixel space
                        #lm.x = int(lm.x * image.shape[1])
                        #lm.y = int(lm.y * image.shape[0])
                        
                        # We append the x and y coordinates of the landmark to the list
                        landmarks.append((lm.x, lm.y))

                # Convert the tuple format to a flat list format
                landmarks_flattened = [item for tup in landmarks for item in tup]

                # Return the landmarks
                return landmarks_flattened
        
        else:
            
            # If no result, we return none
            return None

        

    # Function to load the data from the data folder that creates a json 
    # containing the info extracted from the images/mat files 
    def jsonify(self, scale=True, ext="jpg"):

        # Counter to take track of the number of skipped images due to no results
        skipped = 0

        # Empty dict to store the data that will be converted to json later
        data = {}

        # We iterate through the data folder to get the images and mat files
        for file in tqdm.tqdm(os.listdir(self.datapath)):

            # For each file, we need to create an appropriate entry in the dictionary 
            # which will contain the landmarks and the target angles values.
            # We first ]extract the filename from the path without the extension
            keyname = file.split(".")[0]

            # Then we create an entry in the dictionary with the keyname if keyname doesnt exists,
            # also creating the lists of landmarks and target angles for the inner dictionary
            if keyname not in data:
                data[keyname] = {}
                data[keyname]["landmarks"] = []
                data[keyname]["angles"] = []

            # We analyze only the jpg files avoiding mats, so we cycle N/2 times instead of N
            # (cause we use the jpg filename without ext to open the mat file too)
            if file.endswith(ext):

                # We load the mediapipe face mesh model to extract 
                # the facial landmarks. To do this, we:
                # load the image, flip it to revert camera acquisition changes 
                # and convert it from bgr to rgb
                image = cv2.imread(os.path.join("AFLW2000", file))
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # Then we process the image with the face mesh model
                results = self.get_landmarks(self.mesher.process(image))

                # Check if we got some result, otherwise we skip the image
                if results is None:
                    skipped += 1
                    continue

                # If some results, and the scale flag is true, we scale the landmarks
                if scale:
                    landmarks = self.scaler.scale(results)

                # We finally append the list of landmarks to the data dict
                data[keyname]["landmarks"] = landmarks

                # We then load the associate mat file with pandas; we use the 
                # keyname to get the correct mat file (name of the file but with different ext)
                mat = scipy.io.loadmat(os.path.join("AFLW2000", keyname+".mat"))

                # We extract the yaw, pitch and roll values from the mat file. 
                # NOTE THAT, the values are stored in the mat file in this specific order:
                # - PITCH
                # - YAW
                # - ROLL
                pitch = float(mat["Pose_Para"][0][0])
                yaw = float(mat["Pose_Para"][0][1])
                roll = float(mat["Pose_Para"][0][2])

                # We append the values to the data dict in our order (yaw, pitch, roll)
                data[keyname]["angles"] = [yaw, pitch, roll]

        # Print the number of skipped images
        print("Skipped {} images due to no results on a total of {} images from folder".format(skipped, len(os.listdir(self.datapath))/2))

        # After all the data has been fetched, convert and write JSON object to file
        with open("dataset.json", "w") as outfile: 
            json.dump(data, outfile)


    # Function to load the data into a dataframe ready for training 
    def load(self, scale=True, ext="jpg"): 

        # Counter to take track of the number of skipped images due to no results
        skipped = 0

        # We declare the train data and target lists 
        train_data = []
        target_data = []

        # We iterate through the data folder to get the images and mat files
        for file in tqdm.tqdm(os.listdir(self.datapath)):

            # If the file is in the desired extesion
            if file.endswith(ext):

                # We load the mediapipe face mesh model to extract 
                # the facial landmarks. To do this, we:
                # load the image, flip it to revert camera acquisition changes 
                # and convert it from bgr to rgb
                image = cv2.imread(os.path.join("AFLW2000", file))
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # Then we process the image with the face mesh model
                results = self.get_landmarks(self.mesher.process(image))

                # Check if we got some result, otherwise we skip the image
                if results is None:
                    skipped += 1
                    continue
                
                # If the scale flag is true, we scale the landmarks
                if scale:
                    landmarks = self.scaler.scale(results)

                # If we got some results, we also load the associate mat file 
                mat = scipy.io.loadmat(os.path.join("AFLW2000", file.split('.')[0]+".mat"))

                # We extract the yaw, pitch and roll values from the mat file. 
                # NOTE THAT, the values are stored in the mat file in this specific order:
                # - PITCH
                # - YAW
                # - ROLL
                pitch = float(mat["Pose_Para"][0][0])
                yaw = float(mat["Pose_Para"][0][1])
                roll = float(mat["Pose_Para"][0][2])

                # We append the list of just created landmarks to the train data 
                train_data.append(landmarks)

                # We append the target values to the target data in our order
                target_data.append([yaw, pitch, roll])
        
        # We then convert the lists to pandas dataframe for the ease of consulting
        # (we could have converted to numpy array too)
        train_data = pd.DataFrame(train_data)
        target_data = pd.DataFrame(target_data)

        # Print the number of skipped images
        print("Skipped {} images due to no results on a total of {} images from folder".format(skipped, len(os.listdir(self.datapath))/2))

        # We return it 
        return train_data, target_data

    # Function to make a train test split of the dataset
    def train_test_split(self, X, y, test_size=0.2, random_state=69):
            
            # We split the dataset into x,y train and x,y test set
            return train_test_split(X, y, test_size=test_size, random_state=random_state)
    




                


         
