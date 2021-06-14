from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

folder1 = "C:/Users/Maria/PycharmProjects/IAProject/test/*.png"
folder2 = "C:/Users/Maria/PycharmProjects/IAProject/train/*.png"
folder3 = "C:/Users/Maria/PycharmProjects/IAProject/validation/*.png"
def loadImages(folder):
    images = []
    for filename in glob.glob(folder):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images

test = loadImages(folder1)
test = np.array(test)
train = loadImages(folder2)
train = np.array(train)
validation = loadImages(folder3)
validation = np.array(validation)

def read_label(file):
    lista = []
    for item in file.readlines():
        lista.append(item[11])
    return lista

labelTrain = open("C:/Users/Maria/PycharmProjects/IAProject/train.txt","r")
labelValidation = open("C:/Users/Maria/PycharmProjects/IAProject/validation.txt","r")

arrayLabel1 = read_label(labelTrain)
arrayLabel2 = read_label(labelValidation)

#print(arrayLabel1)

test = test.reshape((5000,3072))
train = train.reshape((30001,3072))
validation = validation.reshape((5000,3072))

plt.plot(test[0])
plt.show()

#svm1 = svm.SVC(C=1.0, kernel='linear')
svm1 = KNeighborsClassifier()
svm1.fit(train, arrayLabel1)
predictions = svm1.predict(validation)
print(metrics.accuracy_score(arrayLabel2, predictions))
print(metrics.f1_score(arrayLabel2, predictions, average="macro"))


predictions = svm1.predict(test)

def showLabel(items, predictions):
    out = open("show.txt","w")
    out.write("id,label\n")
    nr = 0
    for lines in items:
        for i in range(0,11):
            out.write(lines[i])
        out.write(str(predictions[nr]))
        nr += 1
        out.write("\n")
    out.close()
sample = open("C:/Users/Maria/PycharmProjects/IAProject/sample_submission.txt","r")
a = sample.readline()
showLabel(sample.readlines(), predictions)

