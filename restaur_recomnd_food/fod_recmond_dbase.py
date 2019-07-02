#This is code for recommend food if old person come to shop
#if new one come than it print a new person with no recommendation
import face_recognition
import cv2
import numpy as np
import pandas as pd
import time
#video_capture = cv2.VideoCapture('172.20.10.3:8080')
video_capture = cv2.VideoCapture(0)

# In this we read previous data and append the data like face encodings name previous orders
smp_dat=pd.read_csv('details_7.csv')
known_face_encodings=[]#stores face encoding
known_face_names=[]#store names 
food_order=[]# previous food order
for i in range(len(smp_dat)):
    img_name=str(smp_dat['mobileno'][i])
    path=img_name+".jpeg"
    image1 = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image1)[0]

    known_face_encodings.append(encoding)
  
    known_face_names.append(smp_dat['name'])
    food_order.append(smp_dat['foodorder'][i])

#Now we detect the face if in databse
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
flag=0
flag_1=0
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]# BGR color(OpenCV) to RGB color (face_recognition)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Get all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if face matches from any saved images
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                flag=1
            face_names.append(name)
    flag_1+=1
    #If person is not found database than break the loop 
    if flag==0:
        cv2.imshow('Video', frame)
        if flag_1==6:
            print("He is new person")
            break

    else:
        for (a,b,c,d), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            a *= 4
            b *= 4
            c *= 4
            d *= 4
            
            cv2.rectangle(frame, (d,a), (b, c), (0, 0, 255), 2)
        # Display the results    
        print("Hi ",name[best_match_index])
        print("Recommend the food")
        print(smp_dat['foodorder'][best_match_index])
        cv2.imshow('Video', frame)
        break
    if cv2.waitKey(5) & 0xFF == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()
        break
    
'''time.sleep(10)
video_capture.release()
cv2.destroyAllWindows()
'''
