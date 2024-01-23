import posemodels

# This file is created with the sole purpose of making the code more readable and
# easier to maintain. The models library is a collection of classes that implement
# the models that are used in the project. If someone wants to add a new model,
# they can add it to the posemodels library and then add a new if statement in the
# load function to load the model based on the given model name. This way we
# can avoid having a huge if-else statement in the hpe.py file and we can keep
# the code clean and readable.
# The only real requirement is that the new model implemented must have three 
# fundamental functions: train, eval and predict!
# - The train function should train the model based on the given training data (and assign the 
#   best model to its class instance to make predictions; see the model class for further reference).
# - The eval function should evaluate the model based on the given test data (print results).
# - The predict function should return, given an image, the pose parameters and its landmark.
# Thats it!

# Function to load a model from the models library based on the given model name
def load(model_name, mesh_type="mp", mesh_conf=0.5, mesh_iou=0.5):

    # If the model name is svr
    if model_name == "svr":

        # We return a svr model
        return posemodels.Regressor(model_name, mesh_type, mesh_conf, mesh_iou) 
    
    # If the model name is xgboost
    elif model_name == "xgboost":

        # We return a xgboost model
        return posemodels.Regressor(model_name, mesh_type, mesh_conf, mesh_iou)

    # ELIF: Other models can be added here if implemented in the models library
    # elif model_name == "other_model":
    #    return models.OtherModel(model_name_if_needed)