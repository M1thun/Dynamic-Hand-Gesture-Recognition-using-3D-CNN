# Dynamic-Hand-Gesture-Recognition-using-3D-CNN
Hand gesture recognition in computer science and language translation is the means of recognizing hand gestures through mathematical methods. Gesture recognition has become one of growing fields of research. Hand gesture recognition has ample number of applications including human–computer interaction, sign language and virtual/augmented gaming technology. Users can perform gestures to control or interact with devices without physically touching them. There are many architectures designed in the field of gesture detection, but existing traditional solutions are not robust to detect hand gestures with high accuracy in real time in the presence of complex patterns in performing hand gestures. In this paper, we present a fast and efficient algorithm for classifying different dynamic hand gestures using 3D-convolution neural networks. Unlike 2D-convolution neural networks, 3D-convolution networks extract features along the temporal dimension for analysis of gestures performed in videos. The paper also focuses on improving accuracy and describes data preprocessing and optimization techniques for obtaining the model inference in real time at 30fps. Our method achieves a correct recognition accuracy of 90.7% for the evaluation made on the testing videos in Chalearn LAP Continuous Gesture dataset. The detection process can be tested on laptops with standard specifications.

------

### Output Samples
![output_gesture](https://user-images.githubusercontent.com/35320633/95294282-9ef47380-0892-11eb-98c2-dfaed70688c5.jpg)
---
***

### Neural Network Architecture
![Neural-Network](https://user-images.githubusercontent.com/35320633/95294896-b4b66880-0893-11eb-87ed-dba443ba2442.jpg)
---
***

### The repository contains the self-explanatory python3 code starting from:
1. Fetching the data.
2. Dataset generation.
3. Dataset Pre-preprocessing.
4. Define the 3D CNN model.
5. Training the model.
6. Retraining the model.
7. Creating callbacks through tensorboard for graphical visualization.
8. Saving the model as tensorflow protobuf (.pb) file for future usage of mobile deployment.
9. Saving and loading as keras file(.h5).

Last but not the least..

10. ** REAL-TIME visualization of results on Chalearn ConGD videos using OpenCV3. **

---

### Cite as:
Channayanamath M. et al. (2021) Dynamic Hand Gesture Recognition Using 3D-Convolutional Neural Network. In: Satapathy S.C., Bhateja V., Ramakrishna Murty M., Gia Nhu N., Jayasri Kotti (eds) Communication Software and Networks. Lecture Notes in Networks and Systems, vol 134. Springer, Singapore. http://doi-org-443.webvpn.fjmu.edu.cn/10.1007/978-981-15-5397-4_16

---

### Please follow the copyrights procedures for downloading Chalearn Dataset from their official website.
https://gesture.chalearn.org/2016-looking-at-people-cvpr-challenge/isogd-and-congd-datasets
