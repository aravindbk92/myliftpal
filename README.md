# MyLiftingPal
## A Real-Time Weightlifting Form Correction System using a Camera

MyLiftingPal is a vision-based approach to track and correct the form of weightlifters while performing the deadlift exercise. Utilizing a high-resolution camera facing the side of the lifter, the system tracks and evaluates eleven metrics for form and continuously provides correctional audio feedback. In addition, MyLiftingPal tracks the number of correctly completed repetitions and encourages the lifter through audio feedback as and when required. The lifters can naturally interact with the system with the use of hand gestures as these are the only effective means of interacting in a loud and busy gym.

# Requirements
python 3.5  
numpy  
cv2  
simpleaudio  

# How to use
MyLiftingPal works with a feed from your andoird phone. 
- Install IP WebCam Android app on your phone, run it and set its host address in `myliftpal.py`. 
- Then, run `python myliftpal.py`. It will guide you through the lifting process with audio feedback.

# Tests
Individual component tests are inside `./tests`.

# Results
Methodology and results can be found in [docs](docs/myliftingpal-report.pdf).