from deepface import DeepFace
import os
import pandas as pd

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DB_DIRECTORY = os.path.join(CURRENT_DIRECTORY,"user" ,"database")

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]


#face recognition
dfs = DeepFace.find(
  img_path = "photo.jpg", 
  db_path = "user/database", 
  detector_backend = backends[1],
  align = True,
)

print(dfs[0].identity)
