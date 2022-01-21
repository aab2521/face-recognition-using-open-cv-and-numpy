


import cv2
import numpy
cap = cv2.VideoCapture(0)
path = "C:\\Users\\Aabhash\\OneDrive\\Desktop\\opencv-4.0.1\\data\\haarcascades\\haarcascade_frontalcatface.xml"
dirpath = "C:\\Users\\Aabhash\\OneDrive\\Desktop\\face_data"
face_cascade = cv2.CascadeClassifier(path)
face_data = []
labels = []
names = {}
class_id = 0
face_section = np.zeros((100,100),dtype = 'uint8')
for file in os.listdir("C:\\Users\\Aabhash\\OneDrive\\Desktop\\face_data"):
     if file.endswith(".npy"):
            data_item = np.load(dirpath+"\\"+file)
            names[class_id] = file[:-4]
            face_data.append(data_item)
            target = class_id *np.ones((data_item.shape[0],))
            class_id+=1
            labels.append(target)
face_dataset = np.concatenate(face_data,axis=0).reshape((-1,1))
face_labels = np.concatenate(labels,axis=0).reshape(-1,1)
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    face = sorted(faces,key=lambda f:f[2*f[3]])
    for face in faces[-1:]:
        x,y,w,h=face
        face_section = gray_frame[y:y+h,x:x+w]
        face_section = cv2.resize(face_section,(100,100))
        pred= KNN(face_dataset, face_labels, face_section)
        pred_name = name[int(pred)]
        cv2.putText(frame,pred_name,(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),10)
    cv2.imshow("camera",frame)
    key_pressed = cv2.waitKey(1) * 0xFF
    if key_pressed == ord('q'):
       break
cap.release()
cv2.destroyAllWindows() 


# In[ ]:


def distance(x,X):
     return np.sqrt(np.sum((x-X)**2))

def KNN(X,Y,x,K=5):
    m = X.shape[0]
    x = x.reshape((10000,))
    val=[]
    for i in range(m):
        xi=X[i]
         xi = xi.reshape((10000,))
        dist=distance(x,xi)
        val.append((dist,Y[i])) 
    val=sorted(val,key=lambda x:x[0])[:K]
    val=np.asarray(val)
    new_vals= np.unique(val[:,1],return_counts=True)
    index= new_vals[1].argmax()
    output=new_vals[0][index]
    return output   


# In[ ]:




