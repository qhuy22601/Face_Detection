from django.http import JsonResponse
import cv2
from pymongo import MongoClient
from bson.objectid import ObjectId
from django.views.decorators.csrf import csrf_exempt
import json
import os
import numpy as np
from PIL import Image
from datetime import datetime
from django.http import StreamingHttpResponse, HttpResponse,HttpResponseServerError
from django.shortcuts import render
from django.http import HttpResponse
import sqlite3
import hashlib
import requests

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/qhuy/capstone/api_project/api_app/recognizer/trainner.yml')
# cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('/home/qhuy/capstone/api_project/api_app/haarcascade_frontalface_default.xml')


fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)

cam = None

def initialize_camera():
    global cam
    cam = cv2.VideoCapture(0) 

def release_camera():
    global cam
    if cam is not None:
        cam.release()
        cv2.destroyAllWindows()
        cam = None

@csrf_exempt
def view_a(request):
    if request.method == 'POST':
        # # Connect to MongoDB
        # client = MongoClient("mongodb+srv://qhuy:191916823@capstone.l9sjtzd.mongodb.net/test1")
        # db = client["test1"]
        # collection = db["users"]

        # # Generate a new ObjectId for the record
        # id = str(ObjectId())

        # # Get the name and email from the request body
        # body_unicode = request.body.decode('utf-8')
        # body_data = json.loads(body_unicode)
        # name = body_data.get('name', '')
        # email = body_data.get('email', '')

        # # Check if the email already exists in the database
        # existing_record = collection.find_one({"email": email})
        # if existing_record:
        #     # Update the name for the existing record
        #     collection.update_one({"email": email}, {"$set": {"username": name}})
        #     id = existing_record['_id']
        # else:
        #     # Generate a new ObjectId for the record
        #     id = str(ObjectId())
        #     # Insert a new record
        #     collection.insert_one({"_id": id, "username": name, "email": email})

        # # Close the MongoDB connection
        # client.close()

        # sampleNum=0

        # while(True):

        #     ret, img = cam.read()

        #     # Lật ảnh cho đỡ bị ngược
        #     img = cv2.flip(img,1)



        #     # Kẻ khung giữa màn hình để người dùng đưa mặt vào khu vực này
        #     centerH = img.shape[0] // 2;
        #     centerW = img.shape[1] // 2;
        #     sizeboxW = 300;
        #     sizeboxH = 400;
        #     cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
        #                 (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

        #     # Đưa ảnh về ảnh xám
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #     # Nhận diện khuôn mặt
        #     faces = detector.detectMultiScale(gray, 1.3, 5)
        #     for (x, y, w, h) in faces:
        #         # Vẽ hình chữ nhật quanh mặt nhận được
        #         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #         sampleNum = sampleNum + 1
        #         # Ghi dữ liệu khuôn mặt vào thư mục dataSet
        #         cv2.imwrite("/home/qhuy/capstone/api_project/api_app/dataset/User." + id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])

        #     cv2.imshow('frame', img)
        #     # Check xem có bấm q hoặc trên 100 ảnh sample thì thoát
        #     if cv2.waitKey(100) & 0xFF == ord('q'):
        #         break
        #     elif sampleNum>100:
        #         break

        # cam.release()
        # cv2.destroyAllWindows()
        # return JsonResponse({"message": "View A processed successfully"})
        
       
        data = json.loads(request.body)
        id = data.get('id')
        name = data.get('name')
        email = data.get('email')

        sampleNum = 0
        initialize_camera()
        while True:
            ret, img = cam.read()
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(f"/home/qhuy/capstone/api_project/api_app/dataset/User.{id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])

            cv2.imshow('frame', img)

            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 100:
                break

        release_camera()

        conn = sqlite3.connect("/home/qhuy/capstone/api_project/api_app/Face.db")
        cursor = conn.execute("SELECT * FROM Users WHERE id=?", (id,))
        isRecordExist = cursor.fetchone()

        if isRecordExist:
            cmd = "UPDATE Users SET name=? WHERE id=?"
            conn.execute(cmd, (name, id))
        else:
            cmd = "INSERT INTO Users (id, name, email) VALUES (?, ?, ?)"
            conn.execute(cmd, (id, name, email))

        conn.commit()
        conn.close()

        return JsonResponse({"message": "Employee captured and database updated successfully!"})
    else:
        return JsonResponse({"message": "Invalid request method"})

    

# @csrf_exempt
# def api_view_b(request):
#     if request.method == 'POST':
#         path = '/home/qhuy/capstone/api_project/api_app/dataset'
#         faceSamples, Ids = view_b(path)

#         # Train model to extract face features and associate them with each employee
#         # Create a mapping dictionary to encode string IDs as unique integers
#         id_mapping = {id_str: idx for idx, id_str in enumerate(set(Ids))}

#         # Convert the string IDs to integer labels using the mapping dictionary
#         labels = [id_mapping[id_str] for id_str in Ids]

#         # Convert the labels to a numpy array of integer data type
#         labels_array = np.array(labels, dtype=np.int32)

#         # Train the recognizer with the face samples and labels
#         recognizer.train(faceSamples, labels_array)
#         # recognizer.train(faceSamples, np.array(Ids))

#         # Save the model
#         recognizer.save('/home/qhuy/capstone/api_project/api_app/recognizer/trainner.yml')

#         print("Trained!")

#         return JsonResponse({"message": "Training completed successfully"})
#     else:
#         return JsonResponse({'message': 'Invalid request method'})


# def view_b(path):
#     # Lấy tất cả các file trong thư mục
#     imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
#     #create empth face list
#     faceSamples=[]
#     #create empty ID list
#     Ids=[]
#     #now looping through all the image paths and loading the Ids and the images
#     for imagePath in imagePaths:
#         if (imagePath[-3:]=="jpg"):
#             print(imagePath[-3:])
#             #loading the image and converting it to gray scale
#             pilImage=Image.open(imagePath).convert('L')
#             #Now we are converting the PIL image into numpy array
#             imageNp=np.array(pilImage,'uint8')
#             #getting the Id from the image
#             # Id=int(os.path.split(imagePath)[-1].split(".")[1])
#             filename = os.path.splitext(os.path.basename(imagePath))[0]
#             Id = filename.split(".")[1]

#             # extract the face from the training image sample
#             faces=detector.detectMultiScale(imageNp)
#             #If a face is there then append that in the list as well as Id of it
#             for (x,y,w,h) in faces:
#                 faceSamples.append(imageNp[y:y+h,x:x+w])
#                 Ids.append(str(Id))
#     return faceSamples,Ids


# # Lấy các khuôn mặt và ID từ thư mục dataSet
# faceSamples,Ids = view_b('/home/qhuy/capstone/api_project/api_app/dataset')

# # Train model để trích xuất đặc trưng các khuôn mặt và gán với từng nahan viên
# recognizer.train(faceSamples, np.array(Ids))

# # Lưu model
# recognizer.save('/home/qhuy/capstone/api_project/api_app/recognizer/trainner.yml')

# print("Trained!")

@csrf_exempt
def api_view_b(request):
    if request.method == 'POST':
        path = '/home/qhuy/capstone/api_project/api_app/dataset'
        faceSamples, Ids = view_b(path)

        # Create a mapping dictionary to encode string IDs as unique integers
        id_mapping = {id_str: idx for idx, id_str in enumerate(set(Ids))}

        # Convert the string IDs to integer labels using the mapping dictionary
        encoded_labels = [id_mapping[id_str] for id_str in Ids]

        # Train the recognizer with the face samples and encoded integer labels
        for i, faceSample in enumerate(faceSamples):
            recognizer.update([faceSample], np.array([encoded_labels[i]], dtype=np.int32))


        # Save the model
        recognizer.save('/home/qhuy/capstone/api_project/api_app/recognizer/trainner.yml')
        # Save the id_mapping dictionary to a file for future use
        with open('/home/qhuy/capstone/api_project/api_app/id_mapping.json', 'w') as f:
            json.dump(id_mapping, f)

        return JsonResponse({"message": "Training completed successfully"})
    else:
        return JsonResponse({'message': 'Invalid request method'})

def view_b(path):
    # Get all file paths in the directory
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # Create empty face list
    faceSamples = []
    # Create empty ID list
    Ids = []
    # Loop through all image paths and load the IDs and images
    for imagePath in imagePaths:
        if imagePath.endswith(".jpg"):
            # Load the image and convert it to grayscale
            pilImage = Image.open(imagePath).convert('L')
            # Convert the PIL image to a numpy array
            imageNp = np.array(pilImage, 'uint8')
            # Get the ID from the image file name
            filename = os.path.splitext(os.path.basename(imagePath))[0]
            Id = filename.split(".")[1]
            # Extract the face from the training image sample
            faces = detector.detectMultiScale(imageNp)
            # If a face is detected, append it to the list along with its ID
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(str(Id))
    return faceSamples, Ids

# @csrf_exempt
# def api_view_b(request):
#     if request.method == 'POST':
#         path = '/home/qhuy/capstone/api_project/api_app/dataset'  # Update with the correct path to your dataSet directory

#         faceSamples, Ids = view_b(path)

#         recognizer = cv2.face.LBPHFaceRecognizer_create()
#         recognizer.train(faceSamples, np.array(Ids))
#         recognizer.save('/home/qhuy/capstone/api_project/api_app/recognizer/trainner.yml')

#         return JsonResponse({"message": "Training completed successfully"})
#     else:
#         return JsonResponse({'message': 'Invalid request method'})

# def view_b(path):
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     faceSamples = []
#     Ids = []
    
#     for imagePath in imagePaths:
#         if imagePath.endswith(".jpg"):
#             pilImage = Image.open(imagePath).convert('L')
#             imageNp = np.array(pilImage, 'uint8')
#             filename = os.path.splitext(os.path.basename(imagePath))[0]
#             id_parts = filename.split(".")[1:]  # Exclude the first segment 'User'
#             numeric_parts = [part for part in id_parts if part.isdigit()]
#             if numeric_parts:
#                 Id = int("".join(numeric_parts))
#             else:
#                 continue
#             # Id = int(os.path.splitext(os.path.basename(imagePath))[0])
            
#             faces = detector.detectMultiScale(imageNp)
#             for (x, y, w, h) in faces:
#                 faceSamples.append(imageNp[y:y+h, x:x+w])
#                 Ids.append(Id)
    
#     return faceSamples, Ids
# @csrf_exempt
# def api_view_b(request):
#     path = '/home/qhuy/capstone/api_project/api_app/dataset'
#     # Get all file paths in the directory
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     # Create empty face list
#     faceSamples = []
#     # Create empty ID list
#     Ids = []
#     label_dict = {}
#     label_counter = 0
#     # Loop through all image paths and load the IDs and images
#     for imagePath in imagePaths:
#         if imagePath.endswith(".jpg"):
#             # Load the image and convert it to grayscale
#             pilImage = Image.open(imagePath).convert('L')
#             # Convert the PIL image to a numpy array
#             imageNp = np.array(pilImage, 'uint8')
#             # Get the ID from the image file name
#             filename = os.path.splitext(os.path.basename(imagePath))[0]
#             id_parts = filename.split(".")[1:2]  # Get only the second element # Exclude the first segment 'User'
#             # numeric_parts = [part for part in id_parts if part.isdigit()]
#             # if numeric_parts:
#             #     Id = int("".join(numeric_parts))
#             # else:
#             #     continue  # Skip the image if the ID cannot be extracted
#             # Id = id_parts
#             id_str = id_parts[0]
#             # Append the extracted ID to the Ids list
            
#             # Extract the face from the training image sample
#             faces = detector.detectMultiScale(imageNp)
#             # If a face is detected, append it to the list along with its ID
#             for (x, y, w, h) in faces:
#                 faceSamples.append(imageNp[y:y+h, x:x+w])
#                 # Ids.append(Id)
#                 Ids.append(id_str)
        
#     # Ids = np.array(Ids, dtype=np.int32)
#     # Train the model to extract face features and assign them to each employee
#     recognizer.train(faceSamples, np.array(Ids))
    
#     # Save the model
#     recognizer.save('/home/qhuy/capstone/api_project/api_app/recognizer/trainner.yml')
    
#     return HttpResponse("Training completed successfully")

###################################################################################################
# def getImagesAndLabels(path):
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
#     faceSamples = []
#     Ids = []
#     label_dict = {}  # Dictionary to map string IDs to integer labels
#     label_counter = 0

#     for imagePath in imagePaths:
#         if imagePath.endswith(".jpg"):
#             pilImage = Image.open(imagePath).convert('L')
#             imageNp = np.array(pilImage, 'uint8')
#             Id = os.path.split(imagePath)[-1].split(".")[1]

#             # Map string IDs to integer labels
#             if Id not in label_dict:
#                 label_dict[Id] = label_counter
#                 label_counter += 1
#             label = label_dict[Id]

#             faces = detector.detectMultiScale(imageNp)
#             for (x, y, w, h) in faces:
#                 faceSamples.append(imageNp[y:y+h, x:x+w])
#                 Ids.append(label)

#     return faceSamples, Ids

# @csrf_exempt
# def api_view_b(request):
#     faceSamples, Ids = getImagesAndLabels('/home/qhuy/capstone/api_project/api_app/dataset')

#     recognizer = cv2.face.LBPHFaceRecognizer_create()

#     recognizer.train(faceSamples, np.array(Ids, dtype=np.int32))
#     recognizer.save('/home/qhuy/capstone/api_project/api_app/recognizer/trainner.yml')
#     return JsonResponse({'success': True})

# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             time = now.strftime('%I:%M:%S:%p')
#             date = now.strftime('%d-%B-%Y')
#             f.writelines(f'{name}, {time}, {date}'+'\n')


# def getProfile(id):
#     client = MongoClient("mongodb+srv://qhuy:191916823@capstone.l9sjtzd.mongodb.net/test1")
#     db = client["test1"]
#     collection = db["users"]
#     profile = collection.find_one({'_id': ObjectId(str(id))})
#     client.close()
#     return profile

# @csrf_exempt
# def view_c(request):
#     # Khởi tạo camera
#     # cam = cv2.VideoCapture(-1)

#     while True:
#         # Đọc ảnh từ camera
#         ret, img = cam.read()
#         if not ret:
#             continue
#         # Lật ảnh cho đỡ bị ngược
#         img = cv2.flip(img, 1)

#         # Vẽ khung chữ nhật để định vị vùng người dùng đưa mặt vào
#         centerH = img.shape[0] // 2
#         centerW = img.shape[1] // 2
#         sizeboxW = 300
#         sizeboxH = 400
#         cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
#                       (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

#         # Chuyển ảnh về xám
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Phát hiện các khuôn mặt trong ảnh camera
#         faces = detector.detectMultiScale(gray, 1.3, 5)

#         # Lặp qua các khuôn mặt nhận được để hiện thông tin
#         for (x, y, w, h) in faces:
#             # Vẽ hình chữ nhật quanh mặt
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

#             # Nhận diện khuôn mặt, trả ra 2 tham số id: mã nhân viên và dist (độ sai khác)
#             id, dist = recognizer.predict(gray[y:y+h, x:x+w])

#             profile = None

#             # Nếu độ sai khác < 25% thì lấy profile
#             if dist <= 80:
#                 profile = getProfile(id)

#             # Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
#             if profile is not None:
#                 cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+30), fontface, fontscale, fontcolor, 2)
#                 markAttendance(profile[1])
#             else:
#                 cv2.putText(img, "Name: Unknown", (x, y+h+30), fontface, fontscale, fontcolor1, 2)

#         cv2.imshow('Face',img)
#     # Nếu nhấn q thì thoát
#         if cv2.waitKey(1)==ord('q'):
#             break;
#     cam.release()
#     cv2.destroyAllWindows()
        
#     return HttpResponse("Camera released and windows closed.", status=200)

# # def view_c(request):
# #     return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')






def markAttendance(name):
    with open('/home/qhuy/capstone/api_project/api_app/Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'{name}, {time}, {date}' + '\n')


def getProfile(id):
    conn = sqlite3.connect("/home/qhuy/capstone/api_project/api_app/Face.db")
    cursor = conn.execute("SELECT * FROM Users WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# @csrf_exempt
# def view_c(request):
#     def video_stream():
#         while True:
#             ret, img = cam.read()
#             img = cv2.flip(img, 1)

#             centerH = img.shape[0] // 2
#             centerW = img.shape[1] // 2
#             sizeboxW = 300
#             sizeboxH = 400
#             cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
#                           (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             faces = detector.detectMultiScale(gray, 1.3, 5)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

#                 id, dist = recognizer.predict(gray[y:y + h, x:x + w])
#                 profile = None

#                 if dist <= 40:
#                     profile = getProfile(id)

#                 if profile is not None:
#                     cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 30), fontface, fontscale, fontcolor, 2)
#                     markAttendance(profile[1])
#                 else:
#                     cv2.putText(img, "Name: Unknown", (x, y + h + 30), fontface, fontscale, fontcolor1, 2)

#             _, jpeg = cv2.imencode('.jpg', img)
#             frame = jpeg.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#             if cv2.waitKey(1) == ord('q'):
#                 break

#     return HttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')


# @csrf_exempt
# def view_c(request):

#     while True:
#         ret, img = cam.read()
#         img = cv2.flip(img, 1)
#         centerH = img.shape[0] // 2
#         centerW = img.shape[1] // 2
#         sizeboxW = 300
#         sizeboxH = 400
#         cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
#                       (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = detector.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             id, dist = recognizer.predict(gray[y:y+h, x:x+w])
#             profile = None
#             if dist <= 90:
#                 profile = getProfile(str(id))
#             if profile is not None:
#                 cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+30), fontface, fontscale, fontcolor, 2)
#                 markAttendance(profile[1])
#             else:
#                 cv2.putText(img, "Name: Unknown", (x, y+h+30), fontface, fontscale, fontcolor1, 2)

#         _, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         cv2.imshow('Face', img)

#         if cv2.waitKey(1) == ord('q'):
#             break

#     cam.release()
#     cv2.destroyAllWindows()
#     return JsonResponse({'success': True})
#######################################################################
@csrf_exempt
def view_c(request):
    id_mapping = {}  # Initialize an empty dictionary

    # Load the id_mapping from the JSON file
    with open('/home/qhuy/capstone/api_project/api_app/id_mapping.json', 'r') as f:
        id_mapping = json.load(f)
    initialize_camera()
    user_id = None
    try:
        while True:
            ret, img = cam.read()
            if img is None:
                break
            img = cv2.flip(img, 1)
            centerH = img.shape[0] // 2
            centerW = img.shape[1] // 2
            sizeboxW = 300
            sizeboxH = 400
            cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                        (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label, dist = recognizer.predict(gray[y:y+h, x:x+w])
                profile = None
                if dist <= 60:
                    # Reverse mapping from label to string ID
                    id_str = [key for key, value in id_mapping.items() if value == label]
                    if id_str:
                        profile = getProfile(id_str[0])
                if profile is not None:
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+30), fontface, fontscale, fontcolor, 2)
                    user_id = id_str[0]
                    if user_id is not None:
                        break

            if user_id is not None:
                # Call the Spring Boot backend API to check-in the user
                check_in_url = "http://localhost:8080/api/attendance/checkin/" + user_id
                try:
                    response = requests.post(check_in_url)
                    if response.status_code == 200:
                        print("Check-in successful")
                        # cam.release()  # Release the camera after a successful check-in
                        # cv2.destroyAllWindows()
                        return JsonResponse({'success': True})
                    else:
                        print("Check-in failed")
                except requests.exceptions.RequestException as e:
                    print("Error occurred during check-in: " + str(e))
                finally:
                    user_id = None  # Reset user_id to None after the check-in is performed

            for (x, y, w, h) in faces:
                if profile is not None:
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+30), fontface, fontscale, fontcolor, 2)
                else:
                    cv2.putText(img, "Name: Unknown", (x, y+h+30), fontface, fontscale, fontcolor1, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            cv2.imshow('Face', img)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        # Release the camera and close the OpenCV windows even if there's an error
        release_camera()
    return JsonResponse({'success': True})


@csrf_exempt
def view_d(request):
    id_mapping = {}  # Initialize an empty dictionary

    # Load the id_mapping from the JSON file
    with open('/home/qhuy/capstone/api_project/api_app/id_mapping.json', 'r') as f:
        id_mapping = json.load(f)
    initialize_camera()
    user_id = None
    try:
        while True:
            ret, img = cam.read()
            if img is None:
                break
            img = cv2.flip(img, 1)
            centerH = img.shape[0] // 2
            centerW = img.shape[1] // 2
            sizeboxW = 300
            sizeboxH = 400
            cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                        (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label, dist = recognizer.predict(gray[y:y+h, x:x+w])
                profile = None
                if dist <= 60:
                    # Reverse mapping from label to string ID
                    id_str = [key for key, value in id_mapping.items() if value == label]
                    if id_str:
                        profile = getProfile(id_str[0])
                if profile is not None:
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+30), fontface, fontscale, fontcolor, 2)
                    user_id = id_str[0]
                    if user_id is not None:
                        break

            if user_id is not None:
                # Call the Spring Boot backend API to check-in the user
                check_in_url = "http://localhost:8080/api/attendance/testCheckOut/" + user_id
                try:
                    response = requests.post(check_in_url)
                    if response.status_code == 200:
                        print("Check-out successful")
                        # cam.release()  # Release the camera after a successful check-in
                        # cv2.destroyAllWindows()
                        return JsonResponse({'success': True})
                    else:
                        print("Check-out failed")
                except requests.exceptions.RequestException as e:
                    print("Error occurred during check-out: " + str(e))
                finally:
                    user_id = None  # Reset user_id to None after the check-in is performed

            for (x, y, w, h) in faces:
                if profile is not None:
                    cv2.putText(img, "Name: " + str(profile[1]), (x, y+h+30), fontface, fontscale, fontcolor, 2)
                else:
                    cv2.putText(img, "Name: Unknown", (x, y+h+30), fontface, fontscale, fontcolor1, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            cv2.imshow('Face', img)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:            
        release_camera()
    return JsonResponse({'success': True})