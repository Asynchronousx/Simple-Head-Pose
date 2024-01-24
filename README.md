
![simplepose](https://github.com/Asynchronousx/Simple-Head-Pose/assets/38207760/bf52f092-8729-4246-b06b-b10e27dddb97)

Simple Head Pose is a lightweight, modular and expandible framework crafted for the task of the head pose estimation, in which the **euler angles** of the head needs to be predicted given an image. Made with ease of use in mind and ready to use out of the box with minimal effort and lines of code.

![example](https://github.com/Asynchronousx/Simple-Head-Pose/assets/38207760/27ad917a-204d-4c2b-b29f-6ee0d4342bb7)

## Video example
Note that being a synthethic benchmark the performance are lower than a real scenario video.

https://github.com/Asynchronousx/Simple-Head-Pose/assets/38207760/7ec5ee92-0443-4514-b4a8-5f29145ed5e8

## Idea
The chosen approach was carefully selected to prioritize reliability and robustness in various environmental conditions. This is why the framework's pipeline is designed and splitted in such tasks:

1. **Person detection**: This task is performed by running a standard YOLOV5 pretrained model on the image to detect and extract all the person in the scene. This is done because often solely the face detection model may fail in complex environment. 
2. **Face detection**: This task is performed by running a custom YOLOV5 pretrained model on the WIDER face dataset to detect and extract faces from the subimage containing the person, to further increase the accuracy of the face detection step as much as we can.
3. **Landmark Extraction**: This task is performed by running a pretrained mediapipe FaceMesh model on the extracted face image to infer the landmark position on the face. From this step, only the most important landmarks are given back as results. 
4. **Head Pose Estimation**: Given the landmarks, we could use various regressor models (SVR and XGBOOST by default using SKlearn) to perform the computation of the euler angles of the head in the given image. 

Note that, step three and four can be done with **your** module if you want to test the inference with something different. More on that on the *Modules* section.

## Performance
The model pipeline can be run entirely on CPU *or* with the aid of GPU for the heavier tasks, such as person/face detection. Even if **run entirely on CPU** (on a mid range one), it achieves real time, staying within the **7/10FPS** range. When using a **GPU** FPS can reach **30+ with ease.**

## Dataset
The dataset used to train those models is the [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip), a dataset composed of ~ 2000 images and their head pose angles label. The dataset itself have the following structure: 

- **image_n.png**
- **image_n.mat**

As long as the dataset you want to use is formatted like that, you can use every source of data, even yours. More on how to train the models in the *Train* section.

## Usage
For a quick startup: 
```bash
git clone https://github.com/Asynchronousx/Simple-Head-Pose
cd Simple-Head-Pose
conda create --name <env> --file requirements.txt
```
Then, simply create a new file inside the folder (or outside, but remember to specify the correct path) and put into it:

```python
import hpe
model = hpe.SimplePose()
model.load("path_to_your_model")
yaw, pitch, roll = model.predict("path_to_your_image")[0][0]
```
We can then use those **euler angles** as we want. For example, printing it!
```python
print(yaw, pitch, roll)
... -0.14561467791220722 -0.09875710772861299 0.18356302592449872
```
And you're good to go!

### Single Face
Now let's see a more in-depth example when we do have only **one face** into our images. We can simply access the first element of the predictions (the poses list) as we did above and extracting the yaw, pitch and roll values directly like that:

```python
import hpe
model = hpe.SimplePose()
model.load("best_model_svr_23_01_24_17")
yaw, pitch, roll = model.predict("examples/faces_1.png")[0][0]
```
### Multiple faces: 
But, what if we do have multiple faces in an image? We simply access the different poses from the list: iterating or directly!

```python
import hpe
model = hpe.SimplePose()
model.load("best_model_svr_23_01_24_17")
poses = model.predict("examples/faces_1.png")[0]
... access the poses list as you prefer!
```
### Full prediction outputs
You can also fetch all the outputs from the prediction function as follows:

```python
import hpe
model = hpe.SimplePose()
model.load("best_model_svr_23_01_24_17")
poses, landmarks, bbox = model.predict("examples/faces_1.png")
```
Here, we do not access directly to what's returned but instead specify all the returning prediction values (poses, landmarks and bounding boxes) as above.

### Usage out of the box 
If you want to download this repo and immediately displaying results, you can! Just start **main_video.py** if you want to stream results from a video source (i.e: your webcam or a video file, default is your webcam) or if you want to just display it from an image, just run **main_image.py**. If you also want to specify a custom video source or image, just edit the relative code with your desired source/path!

### Further explanation
Here, what we did is import our module then instantiating an empty SimplePose object (by default, we use SVR) and then loading a custom pretrained model (you can find various model in the pretrained folder). Once loaded, we can perform the prediction on a given image (or from a video stream if you prefer). **Predictions** inferred from the model will be a list containing three elements: **head angles, landmarks and bounding box coordinates**. 

We then access the first element (pose angles) and from this again we access the first element (first detected pose) and extracting it right into our three variables: **yaw pitch and roll**. And, **that's it!** You can now use your euler angles as you please. 

Note that, if the *image you passed contains more than one face* and the model successfully recognize and estimate the pose from all of them, the *poses list will contain obviously more than one element*! That's why we access the prediction at *[0][0]*: We access the first returned prediction (poses) at the first fetched pose estimated (if present). Remeber to access them accordingly. More on this on further examples paragraph!

## Displaying Results
Other than simply fetching the head pose estimation, you can also use the model to display relevant information about what the model \*actually\* did. If you'd like to show those results fetched from the predict function, you can use the builtin draw function: 

```python
dest_image = model.draw(src_image, poses, landmarks, bbox, draw_face=True, draw_person=False, draw_lms=False, draw_axis=True)
cv2.imshow("Image", image)
```
![image_screenshot_24 01 2024](https://github.com/Asynchronousx/Simple-Head-Pose/assets/38207760/93758a63-6991-4716-8e89-ae80c29ac32e)




The function draw takes in input the in which you'd like to write onto, and the list of poses and landmarks to draw them (if specified) alongside the bounding boxes of the found person/faces. Then some flags to specify what to actually draw on the image. If you'd like a result as shown in the video example, you can simply leave the function without calling any flags since they're on by default (or call the function as we did above).

## Further usage and examples
As the model itself it's pretty simple to use, you can further inrease the complexity of what to use, how and when. For example, let's instantiate now a simple model as before and extracting all the returning parameters from the model:

```python
import hpe
model = hpe.SimplePose()
model.load("path_to_your_model")
poses, landmarks, bbox = model.predict("path_to_your_image")
```

Here we've returned all the returning outputs of the predict function: 
1. **Poses**, a list that will contains tuples in the format:  [(yaw, pitch and roll)], ..] angles
2. **Landmarks**, a list that will contain coordinate (x,y) ([(x,y), ...] tuples of the most important facial landmarks extracted from the mesher modell
3. **Bbox**: A dictionary containing the bounding box of the person and the face if detected ({0: {person: [x1,y1..], face: [x1, y1..]}, ...}

From there you can use those outputs as you please.

### Specify your model 
You can also specify *which* model to use. This framework comes by default with two model: *SVR and XGBoost*, both regressor written using SKlearn. You can specify which model to use like that: 

```python
import hpe
model = hpe.SimplePose(model_type="svr")
model.load("path_to_your_model")
poses, landmarks, bbox = model.predict("path_to_your_image")
```

Or

```python
import hpe
model = hpe.SimplePose(model_type="xgboost")
model.load("path_to_your_model")
poses, landmarks, bbox = model.predict("path_to_your_image")
```

Simple as that!

### Other parameters
You can also specify the confidence the pretrained models needs to have. Those refer to the yolo and mesh models: 

```python
model = hpe.SimplePose(
    model_type="svr", 
    mesh_type="mp", 
    mesh_conf=0.25, 
    mesh_iou=0.45,
    yolo_conf=0.25,
    yolo_iou=0.45
)
```
Those are the default parameters. Note that, the only thing that cannot be changed there is the mesh_type, since at the time of the writing of this readme is the only one implemented. BUT, you can implement also your face mesher if you don't want to use the MP one! This and more are explained into the *Modules* section.

## Training

If you dont wan't to use the pretrained models, wanna try them on another dataset or simply retrain them with more accurate parameters space, you clearly can do this!
The only thing you need to do is to call the train function instead of the load one, passing as input the folder in which the data is contained: 

```python
import hpe
model = hpe.SimplePose(model_type="svr") ## or model_type="xgboost"
model.train("AFLW2000", save=True, split=0.1, ext="jpg")
poses, landmarks, bbox = model.predict("path_to_your_image")
```
As you can see, the only thing that changed is the line specifying the train function instead of the load one, the rest remains basically the same. Those arguments represent: 
1. Save: if save the model or not into the disk (pretrained folder)
2. Split: Percentual of the train-test split
3. Ext: Extension in which the images are saved into the dataset

Those are the default params, so you can also specify only the train folder if not interested. Note that, the *saved model will be saved into the pretrained folder*. So be sure to have a folder named pretrained!

### Your data!
Do you have any other dataset that you want to use? You can just change the name of the train folder inside the train function and you're good to go! The **only thing that must be respected is the format of the dataset**. Since the dataset loading function is unified, your **data structure must be the same as specified into the *Dataset* paragraph**. To further explain the structure, you must have something like that: 

- **image_0.EXT**
- **image_0.mat**
- **image_1.EXT**
- **image_1.mat**
- ...
- **image_n.EXT**
- **image_n.mat**

Basically, **images and mat needs have the same name** to simplify the data loading. 

## Modules

This simple framework has been tought with the concept of simplyifing modularity and provide expandibility to it. So, you can even *add your own head pose estimator model* or implement your *face mesher* if you don't like some of the modules used! 

### Use your own models 
If you don't like either the SVR or the XGBoost model and would like to try one of your own, don't worry, you can easily add yours!

You need simply to do those things in order: 

1. Add your model to the posemodel.py file. You can do this in two ways: 
- Add a regressor model using sklearn: Since both SVR and XGBoost are multi-output regressor and follow the same code-schema, you can add yours in the **__getinstance** function returning your desired model, and then adding your parameters into the **__getparams** function. 
- Add a totally different model: Just add your model into another class. The only thing your model need to respect is to have three function: train, eval and predict! For further information about how those function should behave, please refer to the brief comment section in the **modelhub.py** file. 

2. Into the file modelhub.py file, add your instance into an elif statement returning your model.

3. Use your model by passing as the model_type parameter your model name (as seen above).

I do hope that's this is simply enough and let you expand this framework however you want and please.

### Head Pose Models Class standards
If you want to specify your custom class for the estimation, you need to be sure to follow this schema: the only real requirement is that the new model implemented must have three fundamental functions: train, eval and predict!
- The train function should train the model based on the given training data (and assign the best model to its class instance to make predictions; see the model class for further reference).
- The eval function should evaluate the model based on the given test data (print results).
- The predict function should return, given an image, the pose parameters and its landmark.

Thats it!

### Use your own mesher
Inside the file **poseutils.py** you can actually find the Mesher class. The mesher class is a wrapper that will return different types of mesher based on the selection choice made at the beginning of the main file (when specifying which mesher you want to use). If you don't feel or can't use the mediapipe mesher for some reason, you can easily add your! All you have to do is to define your mesher class as the following pseudocode:

```python
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
```
and then, in the wrapper mesher class, add your model inside the elif as follow: 

```python
# ELIF: Other meshers can be added here if implemented in the poseutils library
# elif mesher_type == "my_mesher":
#    return my_mesher(parameters if needed)
```
Note that, *is fundamental for your mesher to have a process function which returns a flattened 1d array of [x1,y2,x2,y2..] x,y coordinates of the landmarks you want to use!

### Use your own landmarks
You can also increase or decrease the number of landmark you want to process with the mesher and the models in general. Inside poseutils.py aswell, you can find the **LMinfo** class. To add or remove landmarks you want to use, just edit the landmarks in the init function!

```python
def __init__(self):
        # We assign to the class the following attributes
        self.NOSE = 1
        self.FOREHEAD = 10
        self.LEFT_EYE = 33
        ...
        other landmarks you want to use there
```

## Environment and dependencies
You can find a ready-to-use requirement.txt in this repo to be used with your venv or conda envs. Anyway, the major library used in this framework are: 

- PyTorch
- Opencv-Python
- Sklearn
- Mediapipe

## References
- https://github.com/ultralytics/yolov5
- https://github.com/google/mediapipes
