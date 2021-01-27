# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:40:11 2020

@author: cheikh
"""
#libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random as rand
import keras
import tensorflow_addons as tfa
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from keras_radam import RAdam
from imutils import paths
from bs4 import BeautifulSoup

#list file
#for dirname, _, filenames in os.walk('Dataset'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

#This function will get the coordinates of face given in the annotations file
# the coordinates of lower left corner and upper right corner
def generate_box(obj):  
    xmin = int(float(obj.find('xmin').text))
    ymin = int(float(obj.find('ymin').text))
    xmax = int(float(obj.find('xmax').text))
    ymax = int(float(obj.find('ymax').text))
    
    return [xmin, ymin, xmax, ymax]

#This function will give label assciated with each label and convert them to numbers
def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

#Using in this main function we parse the annotations file and get the objects out from them
# Also we use the above two functions here 
def generate_target(image_id, file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
            
        boxes=np.array(boxes)
        labels=np.array(labels)

        img_id = np.array(image_id)
    # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        return (target,num_objs)
    
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-5
EPOCHS = 20
BS = 32
the_dense = 256
the_dropout = 0.25
the_size_img = 224
dataset_name = "Dataset"


imgs = list(sorted(os.listdir(dataset_name+'/JPEGImages/')))
labels = list(sorted(os.listdir(dataset_name+'/Annotations/')))
print(len(imgs))
print(len(labels))



# Here we use the above functions and save results in lists
targets=[]#store coordinates
numobjs=[]#stores number of faces in each image
#run the loop for number of images we have

i=0
for dirname, _, filenames in os.walk(dataset_name+'/Annotations/'):
    for filename in filenames:
        target, numobj = generate_target(i, os.path.join(dirname, filename))
        targets.append(target)
        numobjs.append(numobj)
        i+=1

i=0
face_images=[]
face_labels=[]

for dirname, _, filenames in os.walk(dataset_name+'/JPEGImages/'):
    for filename in filenames:
        img_path = os.path.join(dirname, filename)
        img = cv2.imread(img_path)
        for j in range(numobjs[i]):
            locs = (targets[i]['boxes'][j])
            img1 = img[int(locs[1]):int(locs[3]), int(locs[0]):int(locs[2])]
            img1 = cv2.resize(img1, (the_size_img, the_size_img))
            img1 = img_to_array(img1)
            img1 = preprocess_input(img1)
            face_images.append(img1)
            face_labels.append(targets[i]['labels'][j])
        i+=1
face_images= np.array(face_images, dtype="float32")
face_labels = np.array(face_labels)

#print(len(face_labels))

unique, counts = np.unique(face_labels, return_counts=True)
#print(dict(zip(unique, counts)))

#Encode the labels in one hot encode form
lb = LabelEncoder()
labels = lb.fit_transform(face_labels)
labels = to_categorical(labels)
#print(labels)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=(the_size_img, the_size_img, 3))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(the_dense, activation="relu")(headModel)
headModel = Dropout(the_dropout)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False




#divide data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(face_images, labels,
    test_size=0.2, stratify=labels, random_state=42)

#Free some space.I did this tep as the notebook was running out of space while training
del targets,face_images,face_labels


# Définition de l'optimizer (avec quelques paramètres qu'il faudra adapter à ses besoins)
#opt = RAdam(total_steps=5000, warmup_proportion=0.1, min_lr=INIT_LR,  name='lr')
"""
opt = tfa.optimizers.RectifiedAdam(
    lr=1e-3,
    total_steps=10000,
    warmup_proportion=0.1,
    min_lr=1e-5,
)
"""
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# Utilisation de l'optimizer dans un model (déjà configuré avant)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS,
	class_weight = {0:5 , 1:1, 2:10})

dataset_name = "VALIDATION"
imgs = list(sorted(os.listdir(dataset_name+'/JPEGImages/')))
labels = list(sorted(os.listdir(dataset_name+'/Annotations/')))
print(len(imgs))
print(len(labels))



# Here we use the above functions and save results in lists
targets=[]#store coordinates
numobjs=[]#stores number of faces in each image
#run the loop for number of images we have

i=0
for dirname, _, filenames in os.walk(dataset_name+'/Annotations/'):
    for filename in filenames:
        target, numobj = generate_target(i, os.path.join(dirname, filename))
        targets.append(target)
        numobjs.append(numobj)
        i+=1

i=0
face_images=[]
face_labels=[]

for dirname, _, filenames in os.walk(dataset_name+'/JPEGImages/'):
    for filename in filenames:
        img_path = os.path.join(dirname, filename)
        img = cv2.imread(img_path)
        for j in range(numobjs[i]):
            locs = (targets[i]['boxes'][j])
            img1 = img[int(locs[1]):int(locs[3]), int(locs[0]):int(locs[2])]
            img1 = cv2.resize(img1, (the_size_img, the_size_img))
            img1 = img_to_array(img1)
            img1 = preprocess_input(img1)
            face_images.append(img1)
            face_labels.append(targets[i]['labels'][j])
        i+=1
face_images= np.array(face_images, dtype="float32")
face_labels = np.array(face_labels)

#Encode the labels in one hot encode form
lb = LabelEncoder()
labels = lb.fit_transform(face_labels)
labels = to_categorical(labels)
#print(labels)

(trainX, testX, trainY, testY) = train_test_split(face_images, labels,
    test_size=0.2, stratify=labels, random_state=42)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

print(predIdxs)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs))

print("[INFO] saving mask detector model...")
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
dt_string = "mask_detector_Adam_" + dt_string
print(dt_string)
model.save("Model/"+ dt_string + ".h5")


f = open("Model/"+ dt_string + ".txt" , "x")
f.write("INIT_LR : "+ str(INIT_LR))
f.write("EPOCHS : "+str(EPOCHS))
f.write("Dense : "+ str(the_dense))
f.write("Dropout : "+str(the_dropout))
f.write("Size img : "+str(the_size_img))
f.close()


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Model/"+ dt_string+".png")
plt.show()
"""
model = keras.models.load_model("Model/"+dt_string + ".h5")

#train the saved model again 
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS,
    class_weight = {0:5 , 1:1, 2:10})


predIdxs = model.predict(testX, batch_size=32)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs))

# # serialize the model to disk
# print("[INFO] saving mask detector model...")

# plot the training loss and accuracy
model.save("Model/"+ dt_string + "_V2.h5")


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Model/"+ dt_string+"_V2.png")
plt.show()
"""