# https://www.learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

# === Step 1 : Set Input and Output Videos

# Import numpy and OpenCV
import numpy as np
import cv2
import os, numpy, PIL, cv2
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
 
# Read input video
cap = cv2.VideoCapture('Video_non-stabilised.mp4');
 
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
 
# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec for output video
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
 
# Set up output video
#out = cv2.VideoWriter('Video_stabilised.mp4', fourcc, fps, (w, h))

isColor = 1
fps     = 25
frameW  = 1920
frameH  = 1080
out = cv2.VideoWriter("video2.avi",-1, fps, (frameW,frameH),isColor)

# === Step 2: Read the first frame and convert it to grayscale

# Read first frame
_, prev = cap.read()
 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

# ===  Step 3: Find motion between frames
# ==> 3.1 Good Features to Track
# ==> 3.2 Lucas-Kanade Optical Flow
# ==> 3.3 Estimate Motion

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32) 
 
for i in range(n_frames-2):
  # Detect feature points in previous frame
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
    
  # Read next frame
  success, curr = cap.read() 
  if not success: 
    break
 
  # Convert to grayscale
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 
 
  # Calculate optical flow (i.e. track feature points)
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 
 
  # Sanity check
  assert prev_pts.shape == curr_pts.shape 
 
  # Filter only valid points
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]
 
  #Find transformation matrix
  m = cv2.estimateAffinePartial2D(prev_pts, curr_pts) #will only work with OpenCV-3 or less
  m= m[0] 
  # Extract traslation
  dx = m[0,2]
  dy = m[1,2]
 
  # Extract rotation angle
  da = np.arctan2(m[1,0], m[0,0])
    
  # Store transformation
  transforms[i] = [dx,dy,da]
    
  # Move to next frame
  prev_gray = curr_gray
 
  print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

# === Step 4: Calculate smooth motion between frames
# ==> Step 4.1 : Calculate trajectory

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0) 

# ==> Step 4.2 : Calculate smooth trajectory

def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  # Define the filter 
  f = np.ones(window_size)/window_size 
  # Add padding to the boundaries 
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
  # Apply convolution 
  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  # Remove padding 
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed 

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  SMOOTHING_RADIUS = 3
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
 
  return smoothed_trajectory

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0) 

# Smooth trajectory using moving average filter
smoothed_trajectory = smooth(trajectory); 

# ==> Step 4.3 : Calculate smooth transforms

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
  
# Calculate newer transformation array
transforms_smooth = transforms + difference

# === Step 5: Apply smoothed camera motion to frames

#  ==> Step 5.1 : Fix border artifacts

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

# === setp5: 

# Reset stream to first frame 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
  
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break
 
  # Extract transformations from the new transformation array
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]
 
  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy
 
  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))
 
  # Fix border artifacts   --> coment this line to see the black border thing
  frame_stabilized = fixBorder(frame_stabilized) 
 
  # Write the frame to the file
  frame_out = cv2.hconcat([frame, frame_stabilized])
 
  # If the image is too big, resize it.
  if(frame_out.shape[1] < 1920): 
    frame_out = cv2.resize(frame_out, (round(frame_out.shape[1]/2), round(frame_out.shape[0]/2)));
   
  cv2.imshow("Before and After", frame_out)
  cv2.waitKey(10)
  #out.write(frame_out)

#out.release()
def make_video():
        outimg=None
        fps=2
        size=None
        is_color=True
        format="XVID"
        outvid='Video_stabilised.avi'

        fourcc = VideoWriter_fourcc(*format)
        vid = None
        nFrames = 129

        for x in range(1, nFrames):
            img = cv2.imread("frame" + str(x-1)+".png")

            if vid is None:
                if size is None:
                    size = img.shape[1], img.shape[0]
                vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        vid.release()
        return vid

make_video()

