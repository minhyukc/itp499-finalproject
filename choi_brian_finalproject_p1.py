# Brian Choi
# ITP 499 Fall 2022
# Final Project
# Problem 1

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.utils import np_utils

# 1. Write code to train a NN model which classifies the Chinese numbers dataset.
df = pd.read_csv('/content/drive/MyDrive/chineseMNIST.csv')

feature_set = df.iloc[:, 0:4096]
label = df.iloc[:, 4096:4098]

# 2. Plot the count of each Chinese number.
plt.figure(figsize=(14,8), dpi=100)
cnt_plt = sns.countplot(x=label['label'].values, data=label['label'])
cnt_plt.set_xlabel('label')
plt.show()

# 3. Visualize 25 random characters from the dataset.
# Be sure that the plot shows both the English number and the Chinese number.
# !wget 'https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf'
# !mv 'SimHei.ttf' /usr/share/fonts/truetype/
fontprop = fm.FontProperties(fname='/usr/share/fonts/truetype/SimHei.ttf')
pixel_lst = []
plt.figure(figsize=(15,15))
for i in range (25):
  random_select = np.random.randint(0, len(feature_set))
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  pixel = feature_set.iloc[random_select, 0:]
  pixel = np.array(pixel).reshape(64, 64)
  pixel_lst.append(pixel)
  plt.imshow(pixel_lst[i], cmap='gray')
  plt.xlabel(str(label.iloc[random_select, 0]))
  plt.title(label.iloc[random_select, 1], fontproperties=fontprop)
plt.show()

# 4. Scale the pixel values
feature_set = feature_set/255

# 5. Partition the dataset into train and test sets.
# Print the shapes of the train and test data sets.

X_train, X_test, Y_train, Y_test = train_test_split(feature_set, label.iloc[:,0], test_size=0.2, random_state=2022, stratify=label.iloc[:,0])

X_train = np.reshape(X_train.to_numpy(), (-1, 64, 64))
X_test = np.reshape(X_test.to_numpy(), (-1, 64, 64))

y_train = pd.get_dummies(Y_train)
y_test = pd.get_dummies(Y_test)

y_test = y_test.to_numpy()

# 6. Build a model of the NN using keras layers. The type, number and hyperparameters of layers is up to you.
# 7. Display the model summary.

n_classes = 15
model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(n_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

h = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

# 10. Plot the loss and accuracy curves for both train and test partitions.
# Loss Curve
plt.figure(figsize=[6,4])
plt.plot(h.history['loss'], 'black', linewidth=2.0)
plt.plot(h.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

# Accuracy Curve
plt.plot(h.history['accuracy'], 'black', linewidth=2.0)
plt.plot(h.history['val_accuracy'], 'green', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()

# 11.	Visualize the predicted and actual image labels (in Chinese characters) for the first 30 images in the dataset.
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

chinese = label.iloc[:, 1]
chinese_labels = chinese.unique()
a = label.drop_duplicates()
a = a.reindex(index=[6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 0, 1000, 2000, 3000, 4000, 5000])
idx_lst = pd.Index(range(len(a)))
a.set_index(idx_lst)

class_dict = {}
for i in range(len(a)):
  class_dict[a.iloc[i][0]] = a.iloc[i][1]

dict_values = list(class_dict.values())

plt.figure(figsize=[10,10])
for i in range(30):
  plt.subplot(5,6,i+1).imshow(X_test[i], cmap='gray')
  plt.subplot(5,6,i+1).set_title('True: {} \nPredict: {}'.format(dict_values[np.where(y_test[i]==1)[0][0]], dict_values[pred_classes[i]]), fontproperties=fontprop)
  plt.subplot(5,6,i+1).axis('off')
plt.show()

# 12.	Visualize 30 random misclassified images. Display the Chinese characters for both the predicted and actual image labels.
failed_indices = []
idx = 0

for i in y_test:
    if i[0] != pred_classes[idx]:
        failed_indices.append(idx)
    idx = idx + 1

plt.figure(figsize=[10,10])
for i in range (30):
    random_select = np.random.randint(0, len(failed_indices))
    failed_index = failed_indices[random_select]
    plt.subplot(5, 6, i+1).imshow(X_test[failed_index], cmap='gray')
    plt.subplot(5, 6, i+1).set_title("True: %s \nPredict: %s" %
                      (dict_values[np.where(y_test[failed_index]==1)[0][0]],
                       dict_values[pred_classes[i]]), fontproperties=fontprop)
    plt.subplot(5, 6, i+1).axis('off')
