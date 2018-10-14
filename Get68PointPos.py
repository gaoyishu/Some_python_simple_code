"""
this file is used for facial landmarks by dlib and opencv3.4.3
python 3.7
"""


import os
import cv2
import dlib
import sys
import numpy as np
import json


"FILE_NAME is the name of output file within pictures information of 68 points locations"

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get68point(in_file, out_file):

    FILE_PATH = in_file
    OUT_PATH = os.path.join(FILE_PATH, out_file)
    PRED_PATH = "shape_predictor_68_face_landmarks.dat"

    with open(OUT_PATH, "w") as pic_dir:
        for filename in os.listdir(in_file):
            if filename.endswith('.jpg'):
                print(filename)
                face_list = []
                # initialization of facial detector
                detector = dlib.get_frontal_face_detector()
                # initialization of predictor with model developed
                predictor = dlib.shape_predictor(PRED_PATH)
                # pics of candidates
                img = cv2.imread(os.path.join(FILE_PATH, filename))
                # trans colors to gray
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # face number detection
                faces = detector(img_gray, 0)
                if len(faces) != 0:
                    for face in range(len(faces)):
                        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[face]).parts()])
                        for idx, pos in enumerate(landmarks):
                            pos = (pos[0, 0], pos[0, 1])
                            print(idx,pos)
                            face_dict = {}
                            face_dict["x"] = pos[0]
                            face_dict["y"] = pos[1]
                            face_dict["index"] = idx + 1
                            face_list.append(face_dict)
                    face_json = json.dumps(face_list,cls=MyEncoder)
                    pic_dir.write(filename + "\t" + face_json + "\n")
                            # cv2.circle(img, pos, 2, color=(0, 255, 0))


if __name__ == '__main__':
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    get68point(input_dir,output_file)
