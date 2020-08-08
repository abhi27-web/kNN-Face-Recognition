import os
import numpy as np
import cv2

dataset_path = "./Data/"

face_data = []
labels = []

for fx in os.listdir(dataset_path):
    #print(fx)
    if fx.endswith(".npy"):
        l= fx.split(".")[0]  
        
        face_item = np.load(dataset_path + fx)
        #print(face_item.shape)
        print("loaded", l)
        
        face_data.append(face_item)
        # appending
        for i in range(face_item.shape[0]):
            labels.append(l)


X = np.concatenate(face_data, axis =0)
Y= np.array(labels)

# KNN CODE

def distance(pA, pB):
    return np.sum((pB - pA)**2)**0.5

def kNN(X, y, x_query, k=5):
    """
    X - > (m , 30000) np array
    y - > (m,) np array
    x_query -> (1, 30000) np array
    k -> scalar unit
    
    do knn for classification    
    """
    m = X.shape[0]
    distances = []
    
    for i in range(m):
        dis = distance(x_query, X[i])
        distances.append((dis, Y[i]))
    
    distances = sorted(distances)
    distances = distances[:k]
    
    distances = np.array(distances)
    labels = distances[:, 1]
    
    uniq_label, counts = np.unique(labels , return_counts = True)

    pred = uniq_label[counts.argmax()]
    
    return pred

# TEST FACE RECOG


cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, frame = cam.read()
    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray_frame)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    face_section = None

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

        offset = 10
        face_section = frame[y-offset : y+h+offset, x-offset : x+w+offset]
        face_section = cv2.resize(face_section , (100,100))

        name = kNN(X, Y, face_section.reshape(-1,))
        cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow("window", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()