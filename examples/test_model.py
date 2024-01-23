import posemodels

# Init a SVR model object
model = posemodels.Regressor("svr")

# Load a model from file
model.load("bestmodel")

# Test the model with images contained into a test folder
model.test("test")

# Or we can simply pass a single image to the predict function: 
# it returns the predicted pose parameters and the landmarks if specified
hpe, lms = model.predict("your_image_path", return_landmarks=True)
