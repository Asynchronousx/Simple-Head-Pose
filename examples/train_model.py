from dataset import DataManager
from posemodels import Regressor

# Declaring the data handler and the model
loader = DataManager("AFLW2000")

# Declaring an svr model
model = Regressor("svr") 

# Load the dataset 
x, y = loader.load()

# Split the dataset into train and test set (default test size is 0.2 and random state is 69)
X_train, X_test, y_train, y_test = loader.train_test_split(x, y)

# Train and evaluate the model, save model is enabled by default
model.train(X_train, y_train)
model.eval(X_test, y_test)

# Test the model with images contained into a test folder; 
# we pass the path of the folder to the test function to check the results
model.test("test")

# Or we can simply pass a single image to the predict function: 
# it returns the predicted pose parameters and the landmarks if specified
angles, lms = model.predict("your_image_path", return_landmarks=True)
