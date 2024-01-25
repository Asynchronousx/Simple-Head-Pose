from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
from datetime import datetime
from sklearn.svm import SVR
from xgboost import XGBRegressor
import mediapipe as mp
import numpy as np
import poseutils
import joblib
import scipy
import cv2
import os

# Regressor class that will wrap various regressor models supported by Sklearn
class Regressor:

    # Init function to load some default objects. Default regressor is SVR
    # And default mesher is mediapipe face mesh
    def __init__(self, regressor_type="svr", mesher_type="mp", mesh_conf=0.25, mesh_iou=0.45):

        # Init an empy instance of the SVR model
        self.__init_model = self.__getinstance(regressor_type)

        # Model path in which to save models
        self.model_path = "trained"

        # Init the model type
        self.model_type = regressor_type

        # Init a mesher object
        self.mesher = poseutils.Mesher("mp", mesh_conf, mesh_iou)

        # We init a scaler object
        self.scaler = poseutils.Scaler()

        # We init a landmarks_info object
        self.landmarks_info = poseutils.LMinfo()

        # Init the random search def params
        self.param_grid = self.__getparams(regressor_type)

    # Function to get the instance of the model based on the passed type
    def __getinstance(self, regressor_type):

        # If the type is SVR, we return an instance of the SVR model
        if regressor_type == "svr":
            return SVR()

        # If the type is XGBoost, we return an instance of the XGBoost model
        if regressor_type == "xgboost":
            return XGBRegressor()

        # Other models that are compatible with sklearn and grid/random search 
        # can be added here with additional if statements as follows
        # if type == "model_name":
        # ..

    def __getparams(self, regressor_type):

        # If the type is SVR, we return the default params for the SVR model
        if regressor_type == "svr":

            params = {
                'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'estimator__C': np.arange(0.6, 0.75, 0.01),
                'estimator__gamma': np.arange(0.09, 0.1, 0.001),
                'estimator__epsilon': np.arange(0.07, 0.08, 0.001),
                'estimator__degree': [2,3,4]
            }

            return params
        
        # If the type is XGBoost, we return the default params for the XGBoost model
        if regressor_type == "xgboost":

            params = {
                "estimator__colsample_bytree": uniform(0.7, 0.3),
                "estimator__gamma": uniform(0, 0.5),
                "estimator__learning_rate": uniform(0.03, 0.3), # default 0.1 
                "estimator__max_depth": randint(2, 6), # default 3
                "estimator__n_estimators": randint(100, 150), # default 100
                "estimator__subsample": uniform(0.6, 0.4)
            }

            return params

        # Other models params can be added here with additional if statements
        # if type == "model_name":
        # ..

    # Function to set the param gris of the randomized search 
    def set_rcparams(self, params):
        self.param_grid = params

    # Function to train the model
    def train(self, X_train, y_train, save=True):

        # Init the multioutput regressor: we need the MOR because we have multiple outputs
        # and a simple SVR wont work with not-flatten outputs (1D values, but we do have N, 3 target)
        mor = MultiOutputRegressor(self.__init_model)

        # Based on the model, we run different type of parametrized searches:
        if self.model_type == "svr": # OR other models that uses grid search
            self.search = GridSearchCV(mor, self.param_grid, scoring='neg_mean_squared_error', verbose=10, n_jobs=-1)
        elif self.model_type == "xgboost": # OR other models that uses randomized search
            self.search = RandomizedSearchCV(mor, self.param_grid, n_iter=20, verbose=10, n_jobs=-1)
        
        # elif: other type of searches can be added here
        # ..

        # Fit the model
        self.search.fit(X_train, y_train)

        # set the best estimator
        self.model = self.search.best_estimator_

        # If save is true, we save the best model found by the grid search 
        if save:
            current_datetime = datetime.now().strftime("%d_%m_%y_%H")
            self.save(os.path.join(self.model_path, "best_model_"+self.model_type+"_"+current_datetime))

    # Function to evaluate the model using the validation set
    def eval(self, X_val, y_val):

        # Print the best parameters of the best fitted model 
        print("--------- EVALUATION RESULTS ---------")
        print("Best parameters found: {}".format(self.search.best_params_))

        # Getting the train rmse from the grid search
        train_rmse = np.sqrt(-self.search.best_score_)
        print("train_rmse: {}".format(np.sqrt(-self.search.best_score_)))

        # Getting the validation rmse by calling the predict function of the model
        # and taking the difference between the predicted and the actual values
        validation_rmse = np.sqrt(mean_squared_error(y_val, self.model.predict(X_val)))
        print("validation_rmse: {}".format(validation_rmse))
        print("--------------------------------------")

    # Function to predict the pose parameters of images contained into a given folder
    def test(self, folder_path):

        # We get the list of images contained into the folder
        # and we iterate through them
        for file in os.listdir(folder_path):

            # If the file is a jpg
            if file.endswith(".jpg"):

                # We load the image
                image = cv2.imread(os.path.join(folder_path, file))

                # We convert the image from bgr to rgb and flip it 
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # We process the image with the face mesh model
                results = self.mesher.process(image)

                # Check if we got some result, otherwise we skip the image
                if results is None:
                    
                    # We print the skipped image name 
                    print("Skipped {} due to lack of results".format(file))
                    continue

                # We scale the landmarks
                landmarks = self.scaler.scale(results)

                # We predict the pose parameters
                predictions = self.model.predict([landmarks])

                # We get the yaw, pitch and roll values from the target mat file
                mat = scipy.io.loadmat(os.path.join(folder_path, file.split('.')[0]+".mat"))
                yaw = float(mat["Pose_Para"][0][1])
                pitch = float(mat["Pose_Para"][0][0])
                roll = float(mat["Pose_Para"][0][2])
                targets = [yaw, pitch, roll]

                # We print the results
                print("--------- Result for Image: {} ---------".format(file))
                print("Predicted: {}".format(predictions[0]))
                print("Target: {}".format(targets))
                print("----------------------------------------") 

    # Function to predict the pose parameters of a single image.
    # The output is a list of 3 values: yaw, pitch and roll and, if 
    # return_landmarks is true, the landmarks of the face too.
    def predict(self, image, return_landmarks=False):

        # We check if the passed image is a string;
        # Else, we do nothing: the image should already be rgb.
        if isinstance(image, str):

            # If so, we need to load it a convert it to rgb
            image = cv2.imread(image)

            # We convert the image from bgr to rgb and flip it 
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Using the mesher object from the poseutil library
        results_lms = self.mesher.process(image)

        # We check if we got some results, otherwise we return None
        if results_lms is None:
            if return_landmarks:
                return None, None
            else:
                return None
      
        # We scale the landmarks if a results is given
        scaled_lms = self.scaler.scale(results_lms)

        # We predict the pose parameters based on the scaled landmarks
        predictions = self.model.predict([scaled_lms])

        # If return_landmarks is true, we return the landmarks too
        if return_landmarks:

            # Return the predicted pose parameters and the not-scaled landmarks
            return predictions[0], results_lms
        
        # Return only the predicted pose parameters otherwise
        return predictions[0]

    # Function that save the model to the disk 
    def save(self, filename):

        # We use the joblib library to save the model given the name  
        joblib.dump(self.model, filename+".joblib")

    # Function to load a model saved from disk
    def load(self, filename):
            
        # We load the model from the disk
        self.model = joblib.load(os.path.join(self.model_path, filename+".joblib"))
