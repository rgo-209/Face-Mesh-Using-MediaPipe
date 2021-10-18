import cv2
import mediapipe as mp
import numpy as np
import sys
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

class MediaPipe_Impl:
    def __init__(self, in_video_path, out_csv_path='test.csv', out_video_path='', min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Function to initialize a MediaPipe_Impl class object.

        Parameters
        ----------
        in_video_path : string
            the path to the video to be used
        out_csv_path : string
            the path to the output csv file path
        out_video_path : string
            the path to the video to be used
        min_detection_confidence : float
            Minimum detection confidence
        min_tracking_confidence : float
            Minimum tracking confidence

        Notes
        -----
        The parameters are passed to the FaceMesh class from MediaPipe.
        For details, see
        https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py

        """

        self.in_video_path = in_video_path
        self.out_video_path = out_video_path
        self.out_csv_path = out_csv_path

        self.mp_face_mesh = mp.solutions.face_mesh
        self.n_landmarks = 468

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.columns = ['frame_num']
        # store coordinates as
        # [frame_num, x_0, y_0, z_0,  x_1, y_1, z_1, .....  x_467, y_467, z_467 ]
        for i in range(468):
            self.columns.append('x_'+str(i))
            self.columns.append('y_'+str(i))
            self.columns.append('z_'+str(i))



    def generate_features(self):
        """
            This function computes the landmarks using mediapipe
            and stores visualizations if required.
        :return:
        """

        # create reader object for input video
        cap = cv2.VideoCapture(self.in_video_path)

        # execute following when you want to create visuals of the video
        if self.out_video_path != '':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            fr_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            fr_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # create writer object for output video
            out = cv2.VideoWriter(self.out_video_path, fourcc, fps, (int(fr_w), int(fr_h)))

        # count number of frames
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # fill the feats array with nans
        feats = np.empty((n_frames, 1 + 3 * self.n_landmarks), dtype=np.float32)
        feats[:] = np.nan

        print("Reading video %s " %self.in_video_path)

        if self.out_video_path != '':
            print("Storing visualizations to %s " % self.out_video_path)

        with self.mp_face_mesh.FaceMesh(min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence) as face_mesh:
            while cap.isOpened():

                # read frames one by one
                success, image = cap.read()
                if not success:
                    break

                # store original frame
                og_img = image

                # get frame number
                pos_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                feats[pos_frames - 1, 0] = pos_frames
                print(pos_frames, end='\r')

                # convert the BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                # read results of the facemesh
                results = face_mesh.process(image)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0].landmark  # writing only results for first detected face
                    for i in range(self.n_landmarks):
                        # write normalized coordinated to the csv file
                        ldmk = face_landmarks[i]
                        feats[pos_frames - 1, 1 + 3 * i] = ldmk.x
                        feats[pos_frames - 1, 2 + 3 * i] = ldmk.y
                        feats[pos_frames - 1, 3 + 3 * i] = ldmk.z

                        # plot points on the original image
                        # and write to output video
                        if self.out_video_path!='':
                            x = int(ldmk.x * fr_w)
                            y = int(ldmk.y * fr_h)
                            og_img[y - 1:y + 2, x - 1:x + 2, :] = 0
                            og_img[y, x, :] = 255

                    # write the frame to output video
                    out.write(og_img)

        print(pos_frames)

        # release resources
        cap.release()
        if self.out_video_path!='':
            print("Stored visualizations to %s successfully !!" % self.out_video_path)
            out.release()

        print("Generated landmarks for the video %s" %self.in_video_path)
        print("Writing facial landmarks for the video to %s " % self.out_csv_path)

        data = pd.DataFrame(feats, columns=self.columns)
        data.to_csv(self.out_csv_path, index=False)
        print("Saved the landmarks for the video to %s  !!" % self.out_csv_path)


if __name__ == '__main__':

    # input video to use for facemesh data
    in_vid_path = 'test_vid.mp4'

    # output video with points plotted on face
    out_vid_path = 'op_test_vid.mp4'

    # csv to store landmarks
    out_csv_path = 'test_data.csv'

    # create object of MediaPipe_Impl class with params
    mediapipe_obj = MediaPipe_Impl(in_vid_path, out_csv_path, out_vid_path)

    # generate features and create visualizations
    mediapipe_obj.generate_features()