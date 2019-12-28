# Run in Spyder

#### Specify Parameters Here ###
# 1. Dataset Name
# dataset_name = 'NWPU-RESISC45' # use this to deploy on original dataset
dataset_name = 'NWPU-RESISC12'   # use this to deploy on provided dataset of 12 classes

# 2. Specify working directory + dataset name
root_path = dataset_name + '/'

# 3. Specify Model A or Model B
model_type = 'model_a' # or 'model_b'
model_to_evaluate = root_path + dataset_name + '_' + model_type + '.model'
#### End Of Specifying Parameters ###

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import itertools
import os

test_path = root_path + 'test/'
train_path = root_path + 'train/'
model_path = root_path + 'models/'
print('Test Set Images Path:', test_path, sep='\n>>> ')
print('Train Set Images Path:', train_path, sep='\n>>> ')

# ## Train and Test Data Generators

def get_target_names(mypath=train_path, model_type=model_type):
  if model_type == 'model_a':
    return ['wetland', 
            'lake', 
            'farmland', 
            'river', 
            'island',
            'desert', 
            'residential', 
            'industrial_area', 
            'commercial_area', 
            'mountain', 
            'forest', 
            'beach']
  if model_type == 'model_b': # raflie messed up when training model b, reset all runtimes before switching models
    return ['beach',
           'farmland',
           'mountain',
           'forest',
           'desert',
           'commercial_area',
           'industrial_area',
           'island',
           'lake',
           'residential',
           'river',
           'wetland']
  from os import listdir
  from os.path import isfile, join
  dirs = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
  return sorted(dirs)
  
target_names = get_target_names()
train_datagen = ImageDataGenerator(rescale=1/255)
if 'NWPU-RESISC45' in train_path:
  batch_size = 256
else:
  batch_size = 128
if model_type == 'model_a':
  im_size = (96, 96)
else:
  im_size = (128, 128)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=im_size,
        batch_size = batch_size,
        classes = target_names,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1/255)

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=im_size,
        batch_size = batch_size,
        classes = target_names,
        class_mode='categorical')

train_samples = train_generator.n
test_samples = test_generator.n

def log_model_history(model_name, epoch, acc, loss, root_path = model_path):
  import datetime
  time = datetime.datetime.now().__str__()
  save_to = root_path +  model_name + '.log'
  with open(save_to, 'a') as log_file:
    for a, l in zip(acc, loss):
      log = ",".join([time, str(epoch), str(a), str(l)])
      log_file.write(log)
      log_file.write('\n')

def new_cnn(img_size, n_classes, model_type=model_type):
  model = Sequential()
  if model_type == 'model_a':
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
  else:
    model.add(Conv2D(16, (5,5), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
  model.add(MaxPooling2D(2, 2))
  model.add(Conv2D(32, (3,3), activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Conv2D(64, (3,3), activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Conv2D(64, (3,3), activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Conv2D(64, (3,3), activation='relu'))
  model.add(MaxPooling2D(2, 2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  if model_type == 'model_b':
    model.add(Dense(128, activation='relu'))
  model.add(Dense(n_classes, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=0.001),
                metrics=['acc'])
  return model


def visualise(y_true, y_pred, n_samples=10):
  pass

def cm_report(y_true, y_pred, target_names = target_names):
  return [
          confusion_matrix(y_true, y_pred), 
          classification_report(y_true, y_pred, target_names = target_names, output_dict=True),
          classification_report(y_true, y_pred, target_names = target_names, output_dict=False, digits=4)
          ]

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False,
                          figsize=(10, 7)):
    """
    Source code: https://stackoverflow.com/a/50386871
    Given a sklearn confusion matrix (cm), make a cm plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    font_size = 15

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=font_size)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=font_size)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=font_size)
    plt.show()

epoch_number = 0
rounds = 5
epochs = 32

choose_epoch = 128
best_model = load_model(model_to_evaluate)

y_true = test_generator.classes[test_generator.index_array]
_y_pred = best_model.predict_generator(test_generator, test_samples // batch_size+1)
y_pred = np.argmax(_y_pred, axis=1)
y_true = test_generator.classes[test_generator.index_array]

cm, cr, cr_str = cm_report(y_true, y_pred)
print(cr_str)

cr_df = pd.DataFrame(cr).transpose()
cr_df.iloc[:len(target_names)].sort_values(by=['f1-score'], ascending=False).to_csv(model_path + '45-classes-f1score.csv')

# list down averages of all metrics
cr_df.loc['weighted avg']

# plot confusion matrix
plot_confusion_matrix(cm, target_names=target_names, normalize=False, figsize=(15,10))


# ## Get Examples of Wrong Classification

n_samples = 5
images, classified = test_generator[0]
classified = [np.argmax(i) for i in classified]

# true vs predicted
classified, y_pred[:n_samples]


correct, wrong = list(), list()
i = 0
while True:
  if classified[i] == y_pred[i]:
    if len(correct) < n_samples:
      correct.append([images[i], classified[i], y_pred[i]])
  else:
    if len(wrong) < n_samples:
      wrong.append([images[i], classified[i], y_pred[i]])
  if len(correct) + len(wrong) == 2 * n_samples:
    break
  i += 1


# #### Images That Are Correctly Classified

# In[ ]:


for i in correct:
  plt.imshow(i[0])
  plt.axis('off')
  plt.show()
  print(target_names[i[1]])


# #### Images Wrongly Classified

# In[46]:


for i in wrong:
  plt.imshow(i[0])
  plt.axis('off')
  plt.show()
  print("Correct class:", target_names[i[1]])
  print("Classified as:", target_names[i[2]])