# Importing the required modules
import math
import random
from cv2 import PCA_DATA_AS_COL, cv2
import hand_detection_module
from data_generation import num_hand
import pickle
from id_distance import calc_all_distance
import time

def check_winner(pc,user):
  result = ""
  if((user == "PAPER" and pc == "ROCK") or (user == "ROCK" and pc == "PAPER" )):
    result = "PAPER"
  elif((user == "SCISSOR" and pc == "ROCK") or (user == "ROCK" and pc == "SCISSOR" )):
    result = "ROCK"
  elif((user == "SCISSOR" and pc == "PAPER") or (user == "PAPER" and pc == "SCISSOR" )):
    result = "SCISSOR"
  else:
    result = "DRAW"
  
  if user == result and pc != result:
    return "USER"
  elif user != result and pc == result:
    return "COMPUTER"
  else: 
    return "DRAW"


# Loading our trained model file for the prediction
model_name = 'hand_model.sav'

# Defining the function to decode 0,1,2 as Paper,Rock,Scissor
def rps(num):
  if num == 0: return 'PAPER'
  elif num == 1: return 'ROCK'
  else: return 'SCISSOR'

# Font to write on line display
font = cv2.FONT_HERSHEY_PLAIN
hands = hand_detection_module.HandDetector(max_hands=num_hand)
model = pickle.load(open(model_name,'rb'))
cap = cv2.VideoCapture(0)

# Start time
t0 = time.time()
prev = ""
p = 0
u = 0
while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    continue
  image, my_list = hands.find_hand_landmarks(cv2.flip(frame, 1),
                                             draw_landmarks=True)
  c = 0
  if my_list:
    height, width, _ = image.shape
    all_distance = calc_all_distance(height,width, my_list)
    pred = rps(model.predict([all_distance])[0])
    pos = (int(my_list[12][0]*height), int(my_list[12][1]*width))
    image = cv2.putText(image,pred,pos,font,1,(0,0,0),2)
    x = int(my_list[12][0]*height)
    y = int(my_list[12][1]*width)
    
    font = cv2.FONT_HERSHEY_SIMPLEX  
    class_pred = ["ROCK","PAPER","SCISSOR"] 

    if(pred != prev):
      prev = pred
      pc_choice = random.choice(class_pred)
      print("User ",pred)
      print("PC ",pc_choice)
      wins = check_winner(pc_choice,pred)
      wf = wins + " Wins"
      image = cv2.putText(image,wf,(150,470),font,1,(0,0,0),2)
      if wins == "USER":
        u = u + 1
      elif wins == "COMPUTER":
        p = p + 1
        
  uf = "User " + str(u)
  pf = "Computer " + str(p)

  image = cv2.putText(image,uf,(50,400),font,1,(0,0,0),2)
  image = cv2.putText(image,pf,(50,450),font,1,(0,0,0),2)
       
  cv2.imshow('Hands', image)

  k = cv2.waitKey(10)
  if k == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()
