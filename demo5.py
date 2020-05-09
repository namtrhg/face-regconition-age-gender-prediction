import numpy as np
import sys
import os
import argparse
import time
import imutils
import cv2

MODEL_MEAN_VALUES = (78.4263377603,87.7689143744,114.895847746)
age_list = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male','Female']
CLASSES = ["background", "chair"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
FREQ_DIV = 5  # frequency divider for capturing training images
RESIZE_FACTOR = 4
NUM_TRAINING = 20
conf_threshold = 0.7
padding = 20

class DectectFace:
    def __init__(self,model,name,age,gender):
        self.face_dir = 'face_data'
        self.face_name = name
        self.face_age = age
        self.face_gender = gender
        self.path = os.path.join(self.face_dir,self.face_name)
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.model = model
        self.count_captures = 0
        self.count_timer = 0
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-c","--confidence",type=float,default=0.5,
                             help="minimum probability to filter weak detections")
        self.args = vars(self.ap.parse_args())
        self.net = cv2.dnn.readNetFromCaffe('face_deploy.prototxt.txt','face_net.caffemodel')

    def capture_training_images(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            self.count_timer += 1
            ret,frame = video_capture.read()
            inImg = np.array(frame)
            outImg = self.process_image(inImg)
            cv2.imshow('Video',outImg)
            # When everything is done, release the capture on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                return

    def process_image(self,inImg):
        frame = cv2.flip(inImg,1)
        #frame = inImg
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
                                     (300,300),(104.0,177.0,123.0))
        self.net.setInput(blob)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        resized_width,resized_height = (112,92)
        if self.count_captures < NUM_TRAINING:
            detections = self.net.forward()
            img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.path) if fn[0] != '.'] + [0])[-1] + 1
            for i in range(0,detections.shape[2]):
                confidence = detections[0,0,i,2]
                if confidence < self.args["confidence"]:
                    continue
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX,startY,endX,endY) = box.astype("int")
                text = "Initializing... {:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cropped = gray[startY:endY,startX:endX]
                resized = cv2.resize(cropped,(resized_width,resized_height))
                cv2.rectangle(frame,(startX,startY),(endX,endY),(255,0,0),2)
                cv2.putText(frame,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),2)
                if self.count_timer % FREQ_DIV == 0:
                    cv2.imwrite('%s/%s.png' % (self.path,img_no),resized)
                    self.count_captures += 1
                    print("Captured image: ",self.count_captures)

        elif self.count_captures == NUM_TRAINING:
            mylines = []
            f = open("age_gender.txt","a")
            f.write(self.face_name + ' ' + self.face_age + ' ' + self.face_gender + '\n')
            # f.close()
            print("Training data captured. Press 'q' to exit.")
            self.count_captures += 1
        return (frame)

    def train_data(self,model):
        imgs = []
        tags = []
        index = 0
        for (subdirs,dirs,files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir,subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path,0))
                    tags.append(int(tag))
                index += 1
        (imgs,tags) = [np.array(item) for item in [imgs,tags]]
        self.model.train(imgs,tags)
        self.model.save(model)
        print("Training completed successfully")
        return

class RecogLBPHFaces:
    def __init__(self):
        self.face_dir = 'face_data'
        self.model = cv2.face_LBPHFaceRecognizer.create()
        self.face_names = []
        self.face_ages = []
        self.face_genders = []
        self.age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt','age_net.caffemodel')
        self.gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt','gender_net.caffemodel')
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-c","--confidence",type=float,default=0.5,
                             help="minimum probability to filter weak detections")
        self.args = vars(self.ap.parse_args())
        self.net = cv2.dnn.readNetFromCaffe('face_deploy.prototxt.txt','face_net.caffemodel')

    def load_trained_data(self):
        names = {}
        key = 0
        for (subdirs,dirs,files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names
        self.model.read('LBPH_trained_data.xml')

    def show_video(self):
        video_capture = cv2.VideoCapture(0)
        while True:
            ret,frame = video_capture.read()
            inImg = np.array(frame)
            outImg,self.face_names = self.process_image(inImg)
            cv2.imshow('Video',outImg)
            # When everything is done, release the capture on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                return

    def process_image(self,inImg):
        frame = cv2.flip(inImg,1)
        #frame = inImg
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
                                     (300,300),(104.0,177.0,123.0))
        self.net.setInput(blob)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        resized_width,resized_height = (112,92)
        detections = self.net.forward()
        persons = []
        for i in range(0,detections.shape[2]):
            confidence = detections[0,0,i,2]
            if confidence < self.args["confidence"]:
                continue
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            (startX,endX) = np.clip(np.array([startX, endX]), 0,gray.shape[1])
            (startY,endY) = np.clip(np.array([startY, endY]), 0,gray.shape[0])
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cropped = gray[startY:endY,startX:endX]
            resized = cv2.resize(cropped,(resized_width,resized_height))
            face_img = frame[startY:endY, startX:endX]
            # path = ("D:\\images")
            # cv2.imwrite(os.path.join(path, 'afterdetect4.jpg'), cropped)
            fblob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            # Predict Gender
            self.gender_net.setInput(fblob)
            gender_preds = self.gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            # print("Gender : " + gender)
            # Predict Age
            self.age_net.setInput(fblob)
            age_preds = self.age_net.forward()
            age = age_list[age_preds[0].argmax()]
            # print("Age Range: " + age)
            overlay_text = "%s-%s" % (gender, age)
            confidence = self.model.predict(resized)
            cv2.rectangle(frame,(startX,startY),(endX,endY),
                          (0,0,255),2)
            if confidence[1] < 100:
                cal = '{:.2f}%'.format(confidence[1])
                person = self.names[confidence[0]]
                detage = []
                detgender = []
                with open("age_gender.txt", "r") as file:
                    for item in file:
                        data = item.split()
                        if data[0] == person:
                            detage = data[1]
                            detgender = data[2]
                #cv2.putText(frame, person, (startX,y), cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
                cv2.putText(frame, 'Info: %s-%s-%s-%s' % (person, detgender, detage, cal), (startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
                print(person, detgender, detage)
            else:
                person = 'Unknown'
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
                cv2.putText(frame, 'Info: %s-%s' % (person, overlay_text),(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
                #cv2.putText(frame,person,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
                print(person, overlay_text)
        return (frame,persons)

def fname():
    print('Enter person profile name (No spaces): ')
    name = input()
    if name.isalpha():
        return name
    else:
        print('Invalid Input')
        del name
        fname()

def fage():
    print('Enter person profile age: ')
    age = input()
    if age.isdigit():
        if int(age) >100:
            print("Invalid input")
            del age
            fage()
        else:
            return age
    else:
        print("Invalid input")
        fage()

def fgender():
    print('Enter person profile gender: ')
    print('1.Male')
    print('2.Female')
    number = input()
    if number == "1":
        gender = "Male"
    elif number == "2":
        gender = "Female"
    return gender

if __name__ == '__main__':
    print('Train or Regconize: ')
    print('1. Train')
    print('2. Regconize')
    number = input()
    if number == "1":
        algorithm = cv2.face_LBPHFaceRecognizer.create()
        data = 'LBPH_trained_data.xml'
        name = fname()
        age = fage()
        gender = fgender()
        trainer = DectectFace(algorithm,name,age,gender)
        trainer.capture_training_images()
        trainer.train_data(data)
    elif number == "2":
        regconizer = RecogLBPHFaces()
        regconizer.load_trained_data()
        regconizer.show_video()
