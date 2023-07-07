import tkinter
from tkinter import *
from tkinter import filedialog
import customtkinter
import cv2
import numpy as np
import glob
import random
import time



screen = Tk()
title = screen.title('Deteksi Helm Motor')
canvas = Canvas(screen, width=500, height=400)
canvas.pack()


def hasil2(file):
    net = cv2.dnn.readNet("yolov3_training_last_2.weights", "yolov3_training_gabungan.cfg")

    classes = []
    with open("classes_gabungan.txt", "r") as f:
        classes = f.read().splitlines()

    images_path = glob.glob(file)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    random.shuffle(images_path)
    for img_path in images_path:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (550, 500))
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        start = time.time()
        outs = net.forward(output_layers)
        end = time.time()
        print("[INFO] Waktu deteksi yolo {:.6f} detik".format(end - start))

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    print(class_id)

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        font = cv2.FONT_HERSHEY_PLAIN
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.8)
        unique, counts = np.unique(class_ids, return_counts=True)
        tambah = 0
        print(confidences)
        akurasi = DoubleVar()    
        akurasi.set(confidences)
        cv2.rectangle(img, (1, 1), (100, 40), (255, 0, 255), 1)
        for i in range(len(counts)):
            cv2.putText(img, str(classes[i]) + " = " + str(counts[i]), (10, 25 + tambah), font, 1, (255, 0, 255), 1)
            tambah = tambah + 15
        print(indexes)
        daftar = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                daftar.append(label)
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.2f}".format(label, confidences[i])
                # cv2.putText(img,f'{classes[i].upper()} {int(counts[i]*50)}%',
                #            (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(color),1)
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
        print(daftar)

        cv2.imshow("Deteksi Helm", img)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    # my_label2 = Label(screen, textvariable=akurasi)
    # my_label2.pack(side='top')
def openFile():
   filename = filedialog.askopenfilename(initialdir="/",
                                         title="Select a File",
                                         filetypes=(("Text files",
                                                     ".jpg"),
                                                    ("all files",
                                                     ".")))
   hasil2(filename)

def video(file1):
    import cv2

    # Load YOLOv3 model
    net = cv2.dnn.readNet("yolov3_training_last_2.weights", "yolov3_training_gabungan.cfg")

    # Load class names
    with open('classes_gabungan.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Define output file
    out_file = 'output.mp4'

    # Define codec and frame rate
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0

    # Open video file
    cap = cv2.VideoCapture(file1)


    # Get video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define output video writer
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    while cap.isOpened():
        # Read frame from video
        ret, frame = cap.read()

        if ret:
            # Create blob from frame
            blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

            # Set input blob for the network
            net.setInput(blob)

            # Get output layers
            layer_names = net.getLayerNames()
            # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

            # Forward pass through network
            outputs = net.forward(output_layers)

            # Process each output layer
            for output in outputs:
                # Process each detection
                for detection in output:
                    # Get class ID, confidence score, and bounding box
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w // 2
                        y = center_y - h // 2

                        # Draw bounding box and label
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f"{classes[class_id]}: {confidence:.2f}"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            # Write frame to output video
            out.write(frame)

            # Display output frame
            cv2.imshow('Deteksi Helm', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release video capture and writer
    cap.release()
    out.release()

    # Close all windows
    cv2.destroyAllWindows()


def openFile1():
   filename1 = filedialog.askopenfilename(initialdir="/",
                                         title="Select a File",
                                         filetypes=(("Video Files",
                                                     ".mp4"),
                                                    ("all files",
                                                     ".")))
   video(filename1)


def hasil():
    net = cv2.dnn.readNet("yolov3_training_last_2.weights", "yolov3_training_gabungan.cfg")

    classes = []
    with open("classes_gabungan.txt", "r") as f:
        classes = f.read().splitlines()

    images_path = glob.glob(r"motor.jpg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    random.shuffle(images_path)
    for img_path in images_path:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (550, 500))
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        start = time.time()
        outs = net.forward(output_layers)
        end = time.time()
        print("[INFO] Waktu deteksi yolo {:.6f} detik".format(end - start))

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    print(class_id)

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        font = cv2.FONT_HERSHEY_PLAIN
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.8)
        unique, counts = np.unique(class_ids, return_counts=True)
        tambah = 0
        print(confidences)
        akurasi = DoubleVar()
        akurasi.set(confidences)
        cv2.rectangle(img, (1, 1), (100, 40), (255, 0, 255), 1)
        for i in range(len(counts)):
            cv2.putText(img, str(classes[i]) + " = " + str(counts[i]), (10, 25 + tambah), font, 1, (255, 0, 255), 1)
            tambah = tambah + 15
        #print(indexes)
        daftar = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                daftar.append(label)
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                text = "{}: {:.2f}".format(label, confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
        print(daftar)

        cv2.imshow("Deteksi Helm", img)
        key = cv2.waitKey(0)

    cv2.destroyAllWindows()
    # my_label2 = Label(screen, textvariable=akurasi)
    # my_label2.pack(side='top')

def realtime():
    # Memasukan nilai YOLO weights dan cfg
    net = cv2.dnn.readNet("yolov3_training_last_2.weights", "yolov3_training_gabungan.cfg")

    # Set minimal confidence dan threshold untuk mengeluarkan nilai maksimal
    confThreshold = 0.5
    nmsThreshold = 0.4

    # Masukan classes nya
    classesFile = "classes_gabungan.txt"
    classes = []
    with open(classesFile, 'r') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Menentukan warna pada bounding boxes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Memuat realtime
    cap = cv2.VideoCapture(0)

    while True:
        # Membaca video per fream
        ret, frame = cap.read()

        # Membuat blob dari fream
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)

        # Input YOLO network
        net.setInput(blob)

        # Jalankan jaringan YOLO
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Extract bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        classIDs = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confThreshold:
                    centerX, centerY, width, height = detection[:4] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Menerapkan nilai maksimal untuk menghilangkan bounding boxes lebih
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

        # mengambar bounding boxes dan labels pada frame
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Nama Fram
        cv2.imshow('Deteksi Helm', frame)

        # Exit klik 'q'
        if cv2.waitKey(1) == ord('q'):
            break


# membuat Logo Atas
p1 = PhotoImage(file = 'logo/logo.png')
screen.iconphoto(False, p1)
# membuat Logo Screen
logo_img = PhotoImage(file='logo/logo.png')
# resize logo
logo_img = logo_img.subsample(2,2)
canvas.create_image(250,75, image=logo_img)


# Button 1
button = customtkinter.CTkButton(master=screen,
                                 command=hasil,
                                 text='Deteksi Helm',
                                 fg_color=("dodger blue", "dodger blue"))
canvas.create_window(250,170,window= button)
#Button 2
button1 = customtkinter.CTkButton(master=screen,
                                    command=openFile,
                                    text='File Foto',
                                    fg_color=("dodger blue", "dodger blue"))
canvas.create_window(250,220,window= button1)
#Button 5
button4 = customtkinter.CTkButton(master=screen,
                                    command=openFile1,
                                    text='File Video',
                                    fg_color=("dodger blue","dodger blue"))
canvas.create_window(250,270,window= button4)
#Button 3
button2 = customtkinter.CTkButton(master=screen,
                                      command=realtime,
                                      text="Deteksi Realtime",
                                      fg_color=("dodger blue", "dodger blue"))
canvas.create_window(250,320,window= button2)
#Button 4
button3 = customtkinter.CTkButton(master=screen,
                                    command=exit,
                                    text='Exit',
                                    fg_color=("dodger blue","dodger blue"))
canvas.create_window(250,370,window= button3)




screen.mainloop()
