import numpy as np
import os
from sklearn import svm
from sklearn.metrics import accuracy_score
import cv2
from skimage.feature import hog
from joblib import dump


#pentru functia de preprocesare imagine
blur_size = (3, 3)
#pentru a obtine rezultate constante la antrenare folosim un seed
seed = 112
#valoarea de binarizare
threshold_val = 100
#valorea de redimensionare imagine
dsize = (30, 80)
#Functia de incarcare a bazei de date
def load_dataset(directory):
    class_dict = {}
    data = []
    for i, dir in enumerate(sorted(os.listdir(directory))):
        class_dict[i] = dir
        for impath in sorted(os.listdir(os.path.join(directory, dir))):
            data_item = {
                "path": os.path.join(directory,dir,impath),
                "label": i
            }
            data.append(data_item)
    return data, class_dict

#Functie de prelucrare date
#primeste o lista de dictionare
# Ex: {'path': 'data/training_data\\0\\10008.png', 'label': 0}

def change(data):
    a_list = []
    labels = []
    for item in data:
        img = cv2.imread(item['path'],cv2.IMREAD_GRAYSCALE)
        res = preprocess(img)
        a_list.append(res)
        labels.append(item['label'])
    x = np.stack(a_list)
    y = np.array(labels)
    return x, y

def preprocess(img):
    blurred = cv2.GaussianBlur(img, blur_size, 0) # smoothing
    im,thre = cv2.threshold(blurred,threshold_val, 255,cv2.THRESH_BINARY_INV) # binarizer
    thre = np.pad(thre,(2, 2),'constant',constant_values=(0,0))
    res = cv2.resize(thre, dsize=dsize, interpolation=cv2.INTER_CUBIC)  # resize
    return res



#functia de extragere trasaturi pentru o imagine si apoi le arunca in modelul svm
def make_feature(img):
    return hog(
        img, orientations=12,
        pixels_per_cell=(14, 14),
        cells_per_block=(1, 1),
        block_norm="L2"
    )


#Incarcarea bazei de date
data_train, class_dict = load_dataset('data/training_data')
data_test, _ = load_dataset('data/testing_data')

print(data_train[1])
x_train,y_train = change(data_train)
x_test, y_test = change(data_test)

X_train_feature = np.array([make_feature(i) for i in x_train], np.float32)
X_test_feature = np.array([make_feature(i) for i in x_test], np.float32)

#X_train_feature.shape, X_test_feature.shape

# Antrenarea propriu-zisa
svm = svm.LinearSVC(C=0.5, random_state=seed)
svm.fit(X_train_feature,y_train)

y_pre = svm.predict(X_test_feature)
acc_svm = accuracy_score(y_test,y_pre)

print("Accuracy SVM:", acc_svm)


# salvarea modelului svm
dump(svm, 'svm_model.joblib')
print("Modelul a fost salvat.")

#pentru incarcare
#svm = load('svm_model.joblib')


