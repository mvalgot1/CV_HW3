import cv2
import mediapipe as mp

drawing = mp.solutions.drawing_utils
draw_styles = mp.solutions.drawing_styles

my_mesh = mp.solutions.face_mesh


captured_video = cv2.VideoCapture(0)
with my_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
  
  while captured_video.isOpened():
    isConnected, img = captured_video.read()
    if not isConnected:
      continue

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        drawing.draw_landmarks(
            image=img,
            landmark_list = face_landmarks,
            connections = my_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = drawing.DrawingSpec(color=(255,0,255),thickness=1,circle_radius=1),
            connection_drawing_spec = draw_styles
            .get_default_face_mesh_contours_style())
    
    cv2.imshow('Computer Vision Face Detection', cv2.flip(img, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
captured_video.release()