import cv2
import numpy as np


# In[2]:


cap = cv2.VideoCapture(0)
xml = "C:\\Users\\Aabhash\\OneDrive\\Desktop\\opencv-4.0.1\\data\\haarcascades\\haarcascade_frontalcatface.xml"
path = xml
face_cascade = cv2.CascadeClassifier(path)
face_data = []
skip = 0  
face_section = np.zeros((100,100),dtype="uint8")
dirpath = "C:\\Users\\Aabhash\\OneDrive\\Desktop\\face_data"


# In[3]:


name = input("Enter Your Name:")
while True:
    ret,frame = cap.read()
    if ret == False:
        continue
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    for face in faces[-1:]:
        x,y,w,h=face
        face_section = gray_frame[y:y+h , x: x+w]
        face_section = cv2.resize(face_section, (100,100))
        cv2.putText(frame,name,(x,y-30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),10)
    cv2.imshow("Camera",frame)
    if skip%10 == 0:
        face_data.append(face_section)
    skip+=1    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(dirpath+'\\'  +name+ '.npy' , face_data)
cap.release()
cv2.destroyAllWindows()


# In[ ]:




