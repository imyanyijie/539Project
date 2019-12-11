# Introduction
This is a web app for hand drawing recognization. The training data we use for the machine learning model is from [Google QuickDraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset). The source code repository is on [Github](https://github.com/imyanyijie/539Project)

# Purpose
Our goal of this project is to build a tool will detect someone’s hand drawing and try to provide the name of the object.
We use the data provided by Google QuickDraw project to train our CNN+RNN model in order to recognize the handdrawn object.

# Take Away
+ Use Python django APIs to separate business logic and web service, and facilitate the collaboration among team members.
+ CNN model was able to reach 79% accuracy after 8 epochs.
+ CNN+RNN model was able to reach 74% accuracy after 3 epochs.
+ CNN+RNN model was able to reach a simular accuracy with less epochs.

Web app:

![Web App](result/ec716838351b12f2a9f626f5c795dae.jpg)

CNN Top 5:

![CNN Top](result/CNNTOP.JPG)

CNN Confusion Matrix:

![CNN Confusion Matrix](result/CNNCM.jpg)
