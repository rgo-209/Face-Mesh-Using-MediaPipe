# Face Mesh Generation Using MediaPipe

This project can be used for generating face mesh data using MediaPipe and 
storing it to a csv and generate an annoated video with the keypoints plotted.


# About MediaPipe Face Mesh 

MediaPipe Face Mesh is a solution that can be used for generating 468 3D landmarks
of the face. Best thing about MediaPipe is, it's ability to use a single camera and
work real-time while being resource efficient. This makes it a great option for 
small devices like mobiles. To know more about MediaPipe and it's other solutions,
checkout their official website [here](https://google.github.io/mediapipe/solutions/face_mesh.html).

# How to use the code

- **Step 1: Installing MediaPipe library**

    For installing MediaPipe, use the following command in your python environment.
    ```shell
    pip install mediapipe
  ```
- **Step 2: Using the main file**
    For generating points using the main file, edit the values of the
    following variables

    ```python
  # path to the input video
  in_vid_path = 'path/of/video/to/use.mp4'
  
  # keep this empty if you don't want to generate visualization
  out_vid_path = 'path/of/video/to/generate.mp4'
    
  # path to csv file for storing landmarks data
  out_csv_path = 'path/to/landmark/output.csv'
  
  # change the value of minimum detection confidence threshold
  min_detection_confidence = 0.5
    

  # change the value of minimum tracking confidence threshold
  min_tracking_confidence = 0.5
  ```    

- **Step 3: Importing the main file**

    If you want to import the class and use it, create an object
    and pass the above values while calling the function generate_features().  