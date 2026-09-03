# My Understanding of Frames, MediaPipe, Landmarks and Gesture Recognition

## 1. What I understand about a Frame

What I understand is that a **frame is one single image from a video**.

When a camera records a video, it is actually capturing many images very quickly, one after another. For example, if a camera is working at 30 FPS, it captures approximately 30 frames every second.

So I can think of a video like this:

```text
Frame 1 → Frame 2 → Frame 3 → Frame 4 → Frame 5 → ...

When these frames are displayed quickly, we see movement.

I also understand that a frame is represented in Python as a NumPy array containing pixel values.

For example:

(height, width, channels)

The channels represent the colors of the image.

OpenCV normally gives us the frame in BGR format, while MediaPipe expects an RGB image, so we need to convert it:

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

Therefore, my understanding is:

A frame is basically one picture captured by the camera, represented as numbers that the computer can process.

2. What I understand about MediaPipe

I understand MediaPipe as a computer vision framework/toolkit that helps us analyze images and videos.

Instead of us having to build everything from zero, MediaPipe provides different tasks that can analyze different parts of an image.

For example:

MediaPipe
   │
   ├── Hand Landmarker
   ├── Face Landmarker
   ├── Pose Landmarker
   ├── Object Detector
   └── Gesture Recognizer

So if I give MediaPipe a frame containing a person, I can choose the appropriate task depending on what I want to detect.

For our classroom monitoring project, we can use things like:

Hand Landmarker to understand where the student's hands are.
Face Landmarker to understand the face and possibly head direction.
Pose Landmarker to understand the student's body position.
Object Detection to identify objects such as phones or other items.
Gesture Recognition to identify specific hand gestures.

I understand that MediaPipe does not simply look at the image and automatically say:

"This student is cheating."

Instead, MediaPipe gives us useful information about what it detects. We can then use that information in our own program to make decisions.

3. What I understand about Landmarks

A landmark is an important point on something that MediaPipe detects.

For example, when MediaPipe detects a hand, it doesn't just tell us:

"There is a hand."

It can give us 21 important points on that hand.

For example:

0  → Wrist
4  → Thumb tip
8  → Index finger tip
12 → Middle finger tip
16 → Ring finger tip
20 → Pinky tip

These points have coordinates such as:

x
y
z

The coordinates tell us where the point is located.

The image coordinates are normally normalized between 0 and 1.

For example:

x = 0.2
y = 0.7

This tells us approximately where the landmark is located in the image.

I understand that the landmarks are important because they convert something visual, such as a hand or body, into numbers that our program can analyze.

For example, instead of saying:

"The student's hand is close to another student's hand."

we can use the coordinates of the hand landmarks and calculate the distance between them.

Therefore:

Landmarks are important points detected by MediaPipe that allow us to represent parts of a person or object using coordinates.

4. How Frames and Landmarks Work Together

I understand the process like this:

Camera
   ↓
Frame
   ↓
BGR → RGB
   ↓
MediaPipe
   ↓
Landmarks / Detections
   ↓
Analyze the coordinates
   ↓
Make a decision

For example, the camera captures a student.

The camera gives us a frame.

OpenCV receives the frame.

We convert BGR to RGB.

We send the RGB frame to MediaPipe.

MediaPipe analyzes the frame and gives us landmarks.

Then our program can analyze those landmarks.

For example:

Student's hand
      ↓
Hand landmarks
      ↓
Get wrist/fingertip positions
      ↓
Calculate distances
      ↓
Check our rules
      ↓
Normal or suspicious?

This is how I understand that a computer can go from simply seeing an image to analyzing a person's movement.

5. What I Understand About Gesture Recognition

I understand that landmarks and gestures are not exactly the same thing.

Landmarks are just the points.

For example:

21 hand landmarks

do not automatically mean:

"Thumbs Up"

Gesture recognition takes the information from the hand landmarks and uses it to determine what gesture the hand is making.

So I understand the process as:

RGB Frame
    ↓
Hand Landmarker
    ↓
21 Hand Landmarks
    ↓
Gesture Recognition
    ↓
Gesture Category

For example:

21 landmarks
     ↓
Analyze their positions
     ↓
Thumb is extended
Other fingers are closed
     ↓
"Thumbs Up"

MediaPipe has some built-in gestures such as:

Closed Fist
Open Palm
Pointing Up
Thumb Down
Thumb Up
Victory
I Love You
None

So I understand gesture recognition as the process of giving meaning to the arrangement of hand landmarks.

6. How This Connects to Our Classroom Monitoring Project

The main thing I have understood is that the classroom monitoring system will not just look at the video like a human does.

It will process the video step by step.

Camera
   ↓
Frames
   ↓
OpenCV
   ↓
RGB Frame
   ↓
MediaPipe
   ↓
Landmarks / Objects / Gestures
   ↓
Geometry + Rules
   ↓
Decision
   ↓
Alert if necessary

For example, if we want to detect suspicious behavior:

Student's hand
      ↓
Hand landmarks
      ↓
Where is the hand?
      ↓
Is it close to another student's hand?
      ↓
Check for several frames
      ↓
Possibly suspicious
      ↓
Trigger alert

Similarly, we could use body or face landmarks:

Face landmarks
      ↓
Head direction
      ↓
Is the student looking toward another student?
      ↓
Check if this continues for several frames
      ↓
Possibly suspicious

So my understanding is that MediaPipe provides the information, while our own decision logic determines what that information means for the classroom monitoring system.

7. The Main Idea I Have Learned

The biggest thing I have understood is:

A frame is the image. MediaPipe analyzes the image. Landmarks give us important points from the image. Gesture recognition interprets the arrangement of hand landmarks into a meaningful gesture. Then our own rules can use this information to make decisions.

I can summarize everything as:

FRAME
"Here is the current picture."

        ↓

MEDIAPIPE
"Let me analyze this picture."

        ↓

LANDMARKS
"Here are the important points."

        ↓

GESTURE RECOGNITION
"Based on these hand points, this looks like a gesture."

        ↓

OUR DECISION LOGIC
"Based on the movement and rules, is this suspicious?"

        ↓

ACTION
"Continue watching or trigger an alert."
8.  I Am Still Worried/Unclear About

How do we track one student over time?

I understand how we can detect landmarks in one frame, but I am still wondering how the system knows that the landmarks in the next frame belong to the same student.

For example:

Frame 1 → Student A
Frame 2 → Student A
Frame 3 → Student A
Frame 4 → Student A

If there are many students in the classroom, how does the system keep track of who is who?

Question

How will we track individual students across multiple frames so that we can analyze their movement over time instead of analyzing every frame independently?

Final Understanding

At this stage, I understand the foundation as:

Frame → MediaPipe → Landmarks → Gesture/Movement Analysis → Decision

The part I understand best is how the camera produces frames, how MediaPipe processes those frames, and how landmarks give us numerical positions.

The part I want to understand better is how we go from these landmarks to reliable suspicious-behavior detection, especially when there are multiple students and normal movements that could look suspicious.
```
