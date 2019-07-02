import face_recognition
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import tensorflow.keras.backend as K
#import shlex e=shelx.split(file_read)
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
K.clear_session()
# Load the model
km = load_model(r'food_detect_model.hdf5',compile = False)

#video_capture = cv2.VideoCapture('172.20.10.3:8080')
video_capture = cv2.VideoCapture(0)
# e is denoted to the list of Food 101 items
e=['Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare', 'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito', 'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake', 'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla', 'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 'Clam chowder', 'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 'Cup cakes', 'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 'Escargots', 'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 'French fries', 'French onion soup', 'French toast', 'Fried calamari', 'Fried rice', 'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 'Grilled cheese sandwich', 'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot and sour soup', 'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 'Lasagna', 'Lobster bisque', 'Lobster roll sandwich', 'Macaroni and cheese', 'Macarons', 'Miso soup', 'Mussels', 'Nachos', 'Omelette', 'Onion rings', 'Oysters', 'Pad thai', 'Paella', 'Pancakes', 'Panna cotta', 'Peking duck', 'Pho', 'Pizza', 'Pork chop', 'Poutine', 'Prime rib', 'Pulled pork sandwich', 'Ramen', 'Ravioli', 'Red velvet cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed salad', 'Shrimp and grits', 'Spaghetti bolognese', 'Spaghetti carbonara', 'Spring rolls', 'Steak', 'Strawberry shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles']
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
       cv2.imshow("person",frame)
       if flag_1==6:
          print("he is a new preson")
          break
   
    # Display the results
    else:
        for (a,b,c,d),name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            a *= 4
            b *= 4
            c *= 4
            d *= 4
         
            
            cv2.rectangle(frame, (d,a), (b, c), (0, 0, 0), 170)
            
        cv2.imshow('Video', frame)
        # Now we detect food before that we alerady removed the face 
        roi=cv2.resize(frame,(299, 299))
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        roi /= 255.       
        pred=km.predict(roi)
        #print(pred)
        max_e=max(pred)
        ind=pred.argmax()
        pred_value=e[ind]
        print(e[ind])
        #add the food in according to customer in database
        smp_dat['foodorder'][best_match_index]=food_order[best_match_index]+","+pred_value
        smp_dat.to_csv('details_7.csv',index=False)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#If the person in new in database than it crate a new database
if flag==0:
    video_capture.release()
    v=cv2.VideoCapture(0)
    fd=cv2.CascadeClassifier(r'haarcascade_frontalface_alt2.xml')

    mobile_no=input("enter ur mobile no.")
    
    def cap():
        flag=1
        while(flag):
            ret,i1=v.read()
            j=cv2.cvtColor(i1,cv2.COLOR_BGR2GRAY)#for gray color detection
            f=fd.detectMultiScale(j)
      
            if len(f)==1:
                for x,y,w,h in f:
                    images=i1[y:y+h,x:x+w]#crop image
                    flag=0
          
                    cv2.imshow('image',images)
                    
                   
                    
                    

        return images
    i_1=cap()
    img_nam=mobile_no+".jpeg"
    cv2.imwrite(img_nam,i_1)
    name=input("enter name")
    
    food_given=input("What food you like to order")
    
        
    df=pd.read_csv('details_7.csv')
    print(df)
     
    #mmake data frame with columns
    df_1 = pd.DataFrame({'mobileno':[mobile_no],
                      'name': [name],
                      'foodorder':[food_given]
                        })

    #append it with previous dataset
    datafram=df.append(df_1, ignore_index=True)
    #save it in dataset
    datafram.to_csv('details_7.csv',index=False)

