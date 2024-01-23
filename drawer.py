import cv2
import numpy as np
import math

# Defining a class that will be used to draw and visualize the results
# on the images

class Drawer:

    def __getinstance(self, image, intype):

        # We check if the image is a string else we assume it is an image
        # and its already loaded and converted to rgb
        if isinstance(image, intype):
                
                # If so, we need to load it a convert it to rgb while flipping it
                # to increase accuracy (also because bbox are calculated on the flipped img)
                image = cv2.imread(image)
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                return image

        # If the image is already a cv mat, we instead simply return it 
        return image


    # Inner function to draw the bounding boxes on the image
    def __draw(self, image, bbox, color=(0, 255, 0), thickness=2):

        # Extract the bounding box coordinates and draw them on the image
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    # Function to draw the bounding boxes on the image
    def draw_bbox(self, image, bbox, color=(255, 200, 20), thickness=2, draw_person=False, draw_face=True):

        # We check the instance of the passed image 
        image = self.__getinstance(image, str)
        
        # We iterate over the bounding boxes dictionary 
        # and draw the bounding boxes on the image
        for key in bbox.keys():

            # Since for each key (0,1,..N) we have a dictionary of bounding boxes
            # for the person and face, we iterate over the dictionary to draw
            # the bounding boxes for both 

            # Check if there are any person key (person can be absent if no detection)
            if "person" in bbox[key]:

                # If draw_person is true, we draw the bounding box around the person
                if draw_person:
                    self.__draw(image, bbox[key]["person"], color, thickness)

                # We check the same for the face if present
                if "face" in bbox[key]:

                    # If draw_face is true, we draw the bounding box around the face
                    if draw_face:
                        self.__draw(image, bbox[key]["face"], color, thickness)

        # Return the flipped image with the bounding boxes drawn
        return cv2.flip(image,1)

    # Function to draw the landmarks on the image
    def draw_landmarks(self, image, landmarks, color=(255, 0, 0), thickness=2):

        # We check the instance of the passed image
        image = self.__getinstance(image, str)

        # We iterate over the landmarks list and draw the landmarks on the image
        for lms in landmarks:
            
            # We check if the landmarks are valid, if not we continue to the next one   
            if lms is not None:

                # We iterate over the landmarks and draw them on the image
                for lm in lms:

                    # We extract the landmarks from the created tuple and draw them on the image
                    x, y = lm

                    # then draw the landmark on the image
                    cv2.circle(image, (int(x), int(y)), thickness, color, thickness)

        # Return the flipped image with the landmarks drawn
        return cv2.flip(image,1)

    # Function to draw the axis on the whom origin is the nose point
    def draw_axis(self, image, pose, landmarks, bbox, axis_size=50):

        # We check the instance of the passed image
        image = self.__getinstance(image, str)

        # For the number of detected landmarks in that image
        for i, lms in enumerate(landmarks):

            # if the landmarks are valid, we continue
            if lms is not None:

                # We extract the nose point from the landmarks
                x, y = lms[0]

                # We draw a circle on the nose point
                # cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), 2)

                # We extract the pose parameters
                yaw, pitch, roll = pose[i]

                # We convert the yaw to negative since the axis are flipped: 
                yaw = -yaw

                # We need now to calculate the rotation matrix R to rotate the axis 
                # around the origin camera parameters, lets say K. We can do this by
                # using the Rodrigues formula to convert the rotation vector to a
                # rotation matrix. We can then use the rotation matrix to rotate the
                # axis points around the origin. 
                rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
                
                # We create a list of points representing the axis (in homogeneous coordinates using 
                # the identity matrix)
                axes_points = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                ], dtype=np.float64)

                # We then use the rotation matrix to rotate the axis points around the origin
                # using the matrix multiplication between the rotation matrix and the axis points
                axes_points = rotation_matrix @ axes_points

                # We then convert the homogeneous coordinates to cartesian coordinates by dividing
                # each coordinate by the last one (z)
                axes_points = (axes_points[:2, :] * axis_size).astype(int)

                # We then displace the axis points by the nose point coordinates
                axes_points[0, :] = axes_points[0, :] + x
                axes_points[1, :] = axes_points[1, :] + y
                
                # We draw the axis on the image
                cv2.line(image, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)    
                cv2.line(image, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)    
                cv2.line(image, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)

                # Then we draw the text on the imageS
                text = ""
                
                # Analyze the pose parameters to see where the user is looking
                # If the pitch is greater than 0.3, the user is looking up
                if pitch > 0.3:
                    text = 'UP'

                # If the pitch is lower than -0.3, the user is looking down
                elif pitch < -0.3:
                    text = 'DOWN'

                # If the yaw is greater than 0.3, the user is looking left
                elif yaw > 0.3:
                    text = 'LEFT'

                # If the yaw is lower than -0.3, the user is looking right
                elif yaw < -0.3:
                    text = 'RIGHT'
                
                # If the user is not looking in any direction, the user is looking forward
                else:
                    text = 'FRONT'

                # We check if the index is present into the bbox dictionary
                if i in bbox: 

                    # We now check if the key face is present in the bbox dictionary
                    if "face" in bbox[i]:

                        # If so, we extract the face bounding box coordinates
                        x1, y1, x2, y2 = bbox[i]["face"]

                        # We draw a rectangle starting from the top left corner of the face
                        # that will contain the text of the direction the user is looking at
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x1)+80, int(y1)-30), (255, 200, 20), -1)

                        # Build the text to display
                        cv2.putText(image, text, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                        # We convert the pitch, yaw and roll to degrees 
                        yaw = math.degrees(math.asin(math.sin(yaw)))
                        pitch = math.degrees(math.asin(math.sin(pitch)))
                        roll = math.degrees(math.asin(math.sin(roll)))
                        
                        # Display the yaw right under to the top right corner of the face
                        cv2.putText(image, "yaw: {:.2f}".format(yaw), (int(x2)+1, int(y1)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                        # Display the pitch under the yaw
                        cv2.putText(image, "pitch: {:.2f}".format(pitch), (int(x2)+1, int(y1)+40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                        # Display the roll under the pitch
                        cv2.putText(image, "roll: {:.2f}".format(roll), (int(x2)+1, int(y1)+65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        # We return a resize image (1280x720)
        return image