import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}


emoji_dist={0:"./emojis/angry.png",2:"./emojis/disgusted.png",2:"./emojis/fearful.png",3:"./emojis/happy.png",4:"./emojis/neutral.png",5:"./emojis/sad.png",6:"./emojis/surpriced.png"}

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
cap1 = cv2.VideoCapture(0)
def show_vid():
    global cap1
    if not cap1.isOpened():
        print("Can't open the camera1")
        exit()

    flag1, frame1 = cap1.read()
    if not flag1:
        print("Cannot read a frame from the camera.")
        return

    # Process the frame and display it in the GUI
    frame1 = cv2.resize(frame1, (600, 500))
    bounding_box = cv2.CascadeClassifier(
        'C:\\Users\\Hasan\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    print("Number of faces detected:", len(num_faces))

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)

        maxindex = int(np.argmax(prediction))
        show_text[0] = maxindex

    # Convert the frame to an ImageTk format and display in the GUI
    img = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    img_pil = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

def show_vid2():
    frame2 = cv2.imread(emoji_dist[show_text[0]])
    pic2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(frame2)
    imgtk2 = ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2 = imgtk2
    lmain2.configure(image=imgtk2)
    lmain3.configure(text=emotion_dict[show_text[0]], font=('arial', 45, 'bold'))
    lmain2.after(10, show_vid2)

# def update_vid():
#     global cap1  # Access the global cap1 variable
#     show_vid()
#     show_vid2()
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap1.release()  # Release the camera
#         root.destroy()
#     else:
#         cap1.release()  # Release the camera after processing frames
#         root.after(10, update_vid)


if __name__ == '__main__':
    # Create the main GUI window
    root = tk.Tk()
    # img = ImageTk.PhotoImage(Image.open("logo.png"))
    heading = Label(root, bg='black')

    heading.pack()
    heading2 = Label(root, text="Photo to Emoji", pady=20, font=('arial', 45, 'bold'), bg='black', fg='#CDCDCD')
    heading2.pack()

    lmain = tk.Label(master=root, padx=50, bd=10)
    lmain2 = tk.Label(master=root, bd=10)
    lmain3 = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=50, y=250)
    lmain3.pack()
    lmain3.place(x=960, y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900, y=350)

    root.title("Photo To Emoji")
    root.geometry("1400x900+100+10")
    root['bg'] = 'black'
    exitbutton = Button(root, text='Quit', fg="red", command=root.destroy, font=('arial', 25, 'bold')).pack(side=BOTTOM)


    # Function to update the video stream and display emoji
    def update_vid():
        global cap1  # Access the global cap1 variable

        show_vid()  # Update the video stream display
        show_vid2()  # Update the emoji display

        root.after(10, update_vid)  # Call update_vid() periodically


    cap1 = cv2.VideoCapture(0)

    # Start the video update process
    update_vid()

    # Start the GUI event loop
    root.mainloop()

    # Release the camera when the application is closed
    cap1.release()
