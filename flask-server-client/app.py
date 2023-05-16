from flask import Flask,render_template,request, redirect, url_for
import mysql.connector
import datetime
from werkzeug.utils import secure_filename
import cv2
import numpy as np
# from PIL import Image
from modeldeepface import *
import json
import os

# this for optimization
# it will takes time (around 30s) when first clasification
dumy = cv2.imread(r"wajahcrop.png")
dumy1 = get_embedding(dumy)

# UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cnvrt_to_json(img_embeding):
    ss = ""
    for i in range(len(img_embeding)):
        st = ', '.join(map(str, map(lambda x: f'"{x}"' if isinstance(x, str) else x, img_embeding[i])))
        st = '[' + st + ']'
        if i + 1 > len(img_embeding)-1:
            ss = ss + '"{}"'.format(i) + ":" + st
            ss = "'{}'".format("{" +ss +"}")
            break
        ss = ss + '"{}"'.format(i) + ":" + st + ","
    return ss

def get_frame_crop(img):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    crop_img = np.zeros
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = faceCascade.detectMultiScale(gray,1.5,5, minSize=(270,270), maxSize=(400,400))
    # faces = faceCascade.detectMultiScale(gray,1.3,5, minSize=(270,270))
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    if len(faces) > 1:
        # return 2 if more than one face is detected
        return 2
    else:
        if len(faces) == 1:
            print("masuk sini")
            cx,cy,cw,ch = 0,0,0,0
            for (x, y, w, h) in faces:
                cx,cy,cw,ch = x,y,w,h

            # height, width, channels = img.shape
            img_height = int(ch/2)
            roi_color = img[cy:cy+img_height, cx:cx+cw]
            roi_gray_left = gray[cy:cy+img_height, cx:int(cx+cw/2)]
            roi_gray_right = gray[cy:cy+img_height, int(cx+cw/2):cx+cw]

            blank_gray = np.zeros((roi_color.shape[0],int(roi_color.shape[1]/2) ), np.uint8)
            gray_collage_left= np.hstack((roi_gray_left,blank_gray))
            gray_collage_right= np.hstack((blank_gray,roi_gray_right))

            eyesleft = eye_cascade.detectMultiScale(gray_collage_left, 1.1, minNeighbors=6, minSize=(30,30))
            eyesright = eye_cascade.detectMultiScale(gray_collage_right, 1.1, minNeighbors=6, minSize=(30,30))
            print(len(eyesleft))
            print(len(eyesright))
            if len(eyesright) == 1 and len(eyesleft) == 1:
                print("masuk mata")
                # center of right eye
                right_eye_center = (
                        int(eyesright[0][0] + (eyesright[0][2]/2)),
                        int(eyesright[0][1] + (eyesright[0][3]/2)))
                # center of left eye
                left_eye_center = (
                    int(eyesleft[0][0] + (eyesleft[0][2] / 2)),
                    int(eyesleft[0][1] + (eyesleft[0][3] / 2)))
                
                #doing eye aligment
                right_eye_x = right_eye_center[0]
                right_eye_y = right_eye_center[1]
                left_eye_x = left_eye_center[0]
                left_eye_y = left_eye_center[1]

                delta_x = right_eye_x - left_eye_x
                delta_y = right_eye_y - left_eye_y
                try:
                    angle=np.arctan(delta_y/delta_x)
                    angle = (angle * 180) / np.pi
                except:
                    angle = 0

                # Width and height of the image
                h, w = img.shape[:2]
                # Calculating a center point of the image
                # Integer division "//"" ensures that we receive whole numbers
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, (angle), 1.0)
                rotated = cv2.warpAffine(img, M, (w, h))
                rotated_gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(rotated_gray,1.2,5)
                # faces = faceCascade.detectMultiScale(
                #     rotated_gray,1.2,5,minSize=(270,270)
                #     )
                print("hello")
                print(len(faces))
                if len(faces) == 1:
                    for (x, y, w, h) in faces:
                        crop_img = rotated[y:y+h, x:x+w]

                    img_emb = get_embedding(crop_img)
                    #return img_emb if sucssess
                    return img_emb
                else:
                    if len(faces) > 1:
                        # return 2 if more than one face is detected
                        return 2
                    else:
                        #return 4 if face not detected
                        return 4
            else:
                #return 3 if eye not detected properly
                return 3
        else:
            #return 4 if face not detected
            return 4
                
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return 'No file part'
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return 'No selected file'
            # return redirect(request.url)
        if file and allowed_file(file.filename):
            #read image file string data
            # filestr = file.read()
            #convert string data to numpy array
            file_bytes = np.frombuffer(file.read(), np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            img_status = get_frame_crop(img)
            if type(img_status) == np.ndarray:
                text_content = request.form['content']
                imgembeding = []
                imgembeding.append(img_status)
                ss = cnvrt_to_json(imgembeding)
                mydb = mysql.connector.connect(
                    host=os.environ['DB_HOST'],
                    port = int(os.environ['DB_PORT']),
                    user=os.environ['DB_USER'],
                    password=os.environ['DB_PASSWORD'],
                    database = os.environ['DB_DATABASE']
                )
                mycursor = mydb.cursor()
                x = datetime.datetime.now()
                sql = "INSERT INTO users (name,created_at,img_emb) VALUES (%s,%s,"+ss+")"
                val = (text_content,x)
                mycursor.execute(sql, val)
                mydb.commit()
                mycursor.close()
                if(mydb.is_connected()):
                    mydb.close()
                img_status = img_status.tolist()
                img_status = np.asarray(img_status)
                print(img_status)
                return "wajah anda telah teregistrasi"
            else:
                if img_status == 2:
                    return "more than one face is detected"
                elif img_status == 3:
                    return "eye not detected properly"
                elif img_status == 4:
                    return "face not detected"


            # return 'something wrong, try again'
        else:
            return 'file not allowed'
        
    else: 
        return render_template('register.html')

@app.route('/recognition', methods=['GET', 'POST'])
def recognition():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            # flash('No file part')
            return 'No file part'
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return 'No selected file'
            # return redirect(request.url)
        if file and allowed_file(file.filename):
            #read image file string data
            # filestr = file.read()
            #convert string data to numpy array
            file_bytes = np.frombuffer(file.read(), np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            img_status = get_frame_crop(img)
            if type(img_status) == np.ndarray:
                # img_embeding are converted to list in database
                # the converted value is not good
                # this need improvements

                mydb = mysql.connector.connect(
                    host=os.environ['DB_HOST'],
                    port = int(os.environ['DB_PORT']),
                    user=os.environ['DB_USER'],
                    password=os.environ['DB_PASSWORD'],
                    database = os.environ['DB_DATABASE']
                )
                mycursor = mydb.cursor()
                sql = "SELECT * FROM users"
                mycursor.execute(sql)
                myresult = mycursor.fetchall()
                for x in myresult:
                    face_dict = json.loads(x[3])
                    narr = np.asarray(face_dict['0'])
                    result = face_verify(narr,img_status)
                    if (result == True):
                        return x[1]
                    # face_dict['0'] 

                if(mydb.is_connected()):
                    mydb.close()
                return "your face is not registered"
            else:
                if img_status == 2:
                    return "more than one face is detected"
                elif img_status == 3:
                    return "eye not detected properly"
                elif img_status == 4:
                    return "face not detected"


            # return 'something wrong, try again'
        else:
            return 'file not allowed'
        
    else: 
        return render_template('recognition.html')
    
if __name__ == '__main__':
   app.run(debug=True)