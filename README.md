# Machine-Learning-Based-Rock-Paper-and-Scissor

This is Machine Learning based Rock Paper and Scissor.

#### Modules used:-
                    1) Mediapipe
                    2) OpenCV
                    
#### Mediapipe:-
MediaPipe offers ready-to-use yet customizable Python solutions as a prebuilt Python package. MediaPipe Python package is available on PyPI for Linux, macOS and Windows.
To install it use 
#### pip install mediapipe

In mediapipe we have a feature to detect the hand which uses 21 points detection. 

#### Idea:-
           1) I have used the basic elementary formula for calculating the distance between the two points. While tracking the hand, 
           mediapip library returns the list of the (x,y) cordinates of all the 21 points. 
           When we use paper, for the 500 iterations I stores the distance between x and y cordinates to prepare the dataset for
           paper option ,for rock and scissor I did the same.
           2) After collecting the data I have tarined my model with the help of SGD classifier.
           3) After all these thing I have used OpenCV for the hand tracking and predicting the paper,scissor and rock for the game.
           
#### Distance calculation :-
                            sqrt((x1-x2)^2 - (y1-y2)^2)

#### Run:-
          To run the file you need to first run the data_generation file
                python data_generation.py
          It will generate the corrdinates for paper, rock and scissor.
          Then you need to run the train_with_sgd file to trian and save your model.
                python train_with_sgd.py
          At last you need to run main.py file to get the game in action.
                python main.py
                
