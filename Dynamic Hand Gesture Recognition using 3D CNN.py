# DYNAMIC HAND GESTURE RECOGNITION USING 3D CONVOLUTION NEURAL NETWORKS

# Install and Import all these necessary dependencies
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.optimizers import SGD, RMSprop,adam
from keras.utils import np_utils, generic_utils
import tensorflow as tf
import theano
import keras
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn import preprocessing
from keras import backend as K
K.set_image_dim_ordering('th')
#from keras.layers.convolution import BatchNormalization
from keras.callbacks import TensorBoard


# Initialize Path
RESULT_PATH = '/workspace/dgx1/keras/40w'
MODEL_NAME = 'saved3d'
MODEL = 'cnn3d'


# Frame specification
img_rows,img_cols,img_depth=30,30,30
print(img_rows,img_cols)


# Data conversion and generation using Opencv3
X_tr=[]
with open('/workspace/dgx1/keras/chae1.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        print(row)
        vid = '/workspace/dgx1/keras/chalearn/1/'+row[3]+'/'+row[0]+'.avi'
        print(vid)
        frames = []
        cap = cv2.VideoCapture(vid)
        fps = 10
        count=0
    print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  
        for s in range (1,int(row[1])):
            ret, frame = cap.read()
        print((s))    
        for k in range(int(row[1]),int(row[2])+1):
                if(count>29):
                    break;           
                ret, frame = cap.read()
                #cv2.imwrite('kang'+str(count)+'.jpg',frame)
                frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
                count=count+1

             
        for l in range (k,(k+30-count)):
            if(count>30):
                break;
            frames.append(gray)
            count=count+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break;
        cap.release()
        cv2.destroyAllWindows()
        input=np.array(frames)
        print(input.shape)
        ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
        print(ipt.shape)
        X_tr.append(ipt)


		
# Convert the frames read into array       
X_tr_array = np.array(X_tr)   
num_samples = len(X_tr_array)
print (num_samples)



# To set class ID for the range for all videos of gestures classes starting from zero
label=np.ones((num_samples,))
label[0:172]= 0
label[172:268]=1
label[268:380]=2
label[380:505]=3
label[505:638]=4
label[638:775]=5
label[775:874]=6
label[874:970]=7
label[970:1076]=8
label[1076:1168]=9
label[1168:1258]=10
label[1258:1348]=11
label[1348:1438]=12
label[1438:1617]=13
label[1617:1808]=14
label[1808:1981]=15
label[1981:2077]=16
label[2077:2172]=17
label[2172:2362]=18
label[2362:2546]=19
label[2546:2683]=20
label[2683:2870]=21
label[2870:2968]=22
label[2968:3065]=23
label[3065:3162]=24
label[3162:3276]=25
label[3276:3377]=26
label[3377:3512]=27
label[3512:3655]=28
label[3655:3749]=29
label[3749:3852]=30
label[3852:3988]=31
label[3988:4087]=32
label[4087:4217]=33
label[4217:4314]=34
label[4314:4419]=35
label[4419:4516]=36
label[4516:4666]=37
label[4666:4815]=38




# Data Pre-processing and training parameters initialization
train_data = [X_tr_array,label]
(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', y_train.shape)
img_depth=30
print(y_train)
train_set = np.zeros((num_samples, 1, img_rows,img_cols,img_depth))
for h in range(num_samples):
    train_set[h][0][:][:][:]=X_train[h,:,:,:]   
patch_size = 30    # img_depth or number of frames used for each video
print(train_set.shape, 'train samples')



# Initialing the Training Parameters
batch_size =3
nb_classes = 39
nb_epoch =1000


# Converting class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)



# Number of convolutional filters to use at each layer
nb_filters = [32,32]
# Level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]
# Level of convolution to perform at each layer (CONV x CONV)
nb_conv = [5,5]



# Data Pre-processing
train_set = train_set.astype('float32')
train_set -= np.mean(train_set)
train_set /=np.max(train_set)



# Defining the 3D Convolution Neural Network model
model = Sequential()
model.add(Convolution3D(nb_filters[0],kernel_dim1=nb_conv[0],kernel_dim2=nb_conv[0],kernel_dim3=nb_conv[0],input_shape=(1, img_rows, img_cols, patch_size),data_format='channels_first', activation='relu',border_mode='same'))
model.add(BatchNormalization())
model.add(Convolution3D(nb_filters[0],kernel_dim1=nb_conv[0],kernel_dim2=nb_conv[0],kernel_dim3=nb_conv[0],input_shape=(1, img_rows, img_cols, patch_size), activation='relu',border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0]),border_mode='same'))

model.add(Dropout(0.25))

model.add(Convolution3D(64,kernel_dim1=nb_conv[0],kernel_dim2=nb_conv[0],kernel_dim3=nb_conv[0],activation='relu',border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0]),border_mode='same'))

model.add(Dropout(0.25))

model.add(Convolution3D(128,kernel_dim1=nb_conv[0],kernel_dim2=nb_conv[0],kernel_dim3=nb_conv[0],activation='relu',border_mode='same'))
model.add(BatchNormalization())
model.add(Convolution3D(128,kernel_dim1=nb_conv[0],kernel_dim2=nb_conv[0],kernel_dim3=nb_conv[0],activation='relu',border_mode='same'))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0]),border_mode='same'))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, init='normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Run the following five lines of code for retraining the trained model for with increased number of gesture classes 
# by popping and replacing the final softmax layer with more output units or directly compile to train from scratch
model=load_model('./20gweights/saved3d.h5')
model.pop()
model.pop()
for layer in  load.layers:
    layer.name=layer.name + str("_")
	

model.add(Dense(nb_classes,init='normal'))
model.add(Activation('softmax'))
opt=adam(lr=0.0001)


# Compile the model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])



# Spliting the dataset for testing and training
X_train_new, X_val_new, y_train_new,y_val_new =  train_test_split(train_set, Y_train, test_size=0.2, random_state=4)



# Training  the model along with creating callbacks to tensorboard for graphical visualization of training process
tbcallback = keras.callbacks.TensorBoard(log_dir='/workspace/dgx1/keras/40g', histogram_freq=0, write_graph=True, write_images=True)
hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new,y_val_new),
          batch_size=batch_size,nb_epoch = nb_epoch,shuffle=True,verbose=1,callbacks=[tbcallback])
		  
		  
		  
# To convert into tensorflow pb file for future usage
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):  
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph



# Reading the graphdef file
from keras import backend as K
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, "./", "my_model2.pb", as_text=False)




# Loading graphDef file
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    # load model from pb file
    with gfile.FastGFile('my_model2.pb','rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)

		
		
# Saving the trained model to hard disk		
with open(os.path.join(RESULT_PATH, MODEL_NAME) + '.json', 'w')as model_file:
        model_file.write(model.to_json())
model.save(os.path.join(RESULT_PATH, MODEL_NAME) + '.h5')
print('Saved model successfully')



# Loading a trained model from hard disk
new_model=load_model(os.path.join(RESULT_PATH, MODEL_NAME) + '.h5')



# Evaluate the model
score = new_model.evaluate(X_val_new, y_val_new, batch_size=batch_size,verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])





# Testing the videos from Chalearn ConGD dataset on the trained model and printing the result on video using OpenCV3
# Note: We have set random names for five classes and which are printed on the video
# The resulting gesture ID or the label for all 39 gesture classes can be obtained as output
frames = []
cap = cv2.VideoCapture('dataset/noball.avi')
fps = 30
for k in range(30):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

input=np.array(frames)
ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
ipt = np.expand_dims(ipt, axis=0)
ipt = np.expand_dims(ipt, axis=1)
#print(ipt.shape)
ipt=np.array(ipt)

prediction=(loaded_model.predict(ipt))
print(prediction)
label = np.argmax(prediction)
print(label)

print('The action is:')
if label==0:
    text = ("Index")
elif label==1:
    text = ("Shoulder")
elif label==2:
    text = ("Chin")
elif label==3:
    text = ("Super")
elif  label==4:
    text = ("Wave")
else:
    text = ("No Action Detected")
	
# Real-time visualization of the output using OpenCV3
while True:
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame=cv2.resize(frame,(400,240),interpolation=cv2.INTER_AREA)
    cv2.putText(frame,text,(10,25), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('Frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()




# Plotting the results
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(1000)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available) # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
plt.style.use(['classic'])

loss,acc =new_model.evaluate(X_val_new, y_val_new, batch_size=batch_size,verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
