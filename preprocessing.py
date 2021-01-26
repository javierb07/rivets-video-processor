# -*- coding: utf-8 -*-xxxxx
"""
@author: Javier Belmonte
"""
# Import necessary libraries
import cv2
import time
import collections
import os
import numpy as np
import math
import copy
from pathlib import Path
from scipy.stats import linregress
# Class definition
class Preprocessing:
    # Initialization method
    def __init__(self,name,sizex=256,sizey=128,timeStep=1,fps=60,robotSpeed=0.22,frameLength=0.2,histCon=False,alpha=0.8,beta=5):
        self.name = name                                    # Name of the video
        self.fps = fps                                      # Frames per second of the video
        self.sizex = sizex                                  # Desired horizontal pixels of extracted frames
        self.sizey = sizey                                  # Desired vertical pixels of extracted frames
        self.robotSpeed = robotSpeed                        # Robot speed in feets per second
        self.frameLength = frameLength                      # Length of the frame captured by the robot
        self.timeStep = self.frameLength/self.robotSpeed    # Time interval to extract frames
        self.frameStep = timeStep * fps                     # Number of frames elapsed before one is extracted
        self.histCon = histCon                              # Determine if histrogram equalization is going to be used to enhance contrast
        self.alpha = alpha                                  # Alpha value for contrast enhancing 
        self.beta = beta                                    # Beta value for contrast enhancing
    # Method to get input from the user
    def GetInput(self):
        i = ""  
        d = {"Normal":[False,0], "Grayscale": [False,1], "Cropped":[False,2], "Contrast":[False,3], "BlackWhite":[False,4], "Resized":[False,5], "Output":[True,6],"Target":[True,7],"Rivets":[True,8]}     #Default values
        dOrd = collections.OrderedDict(sorted(d.items(), key=lambda t: t[1][1])) # Create an ordered dictionary
        while( i !="Y" or i !="N"): #Handle user input
            i = input("Press ´Y´ to use default settings (extracts only final result) or ´N´to enter custom seetings. ").upper()  #Use upper to make sure that upper and lower case entries are handled             
            if (i == "Y"):
                return dOrd
            elif ( i == "N"):
                n = input("Do you want to save the normal frame? Press Y for yes, any other for no. ").upper()
                if n == "Y":
                    d["Normal"][0] = True
                g = input("Do you want to save the grayscale frame? Press Y for yes, any other for no. ").upper()
                if g == "Y":
                    d["Grayscale"][0] = True
                c = input("Do you want to save the enhanced contrast frame? Press Y for yes, any other for no. ").upper()
                if c == "Y":
                    d["Contrast"][0] = True
                b = input("Do you want to save the black and white frame? Press Y for yes, any other for no. ").upper()
                if b == "Y":
                    d["BlackWhite"][0] = True
                r = input("Do you want to save the resized frame? Press Y for yes, any other for no. ").upper()
                if r == "Y":
                    d["Resized"][0] = True
                cr = input("Do you want to save the cropped frame? Press Y for yes, any other for no. ").upper()
                if cr == "Y":
                    d["Cropped"][0] = True
                done = input("Are you satisfied with your input? Press Y for yes, any other for no. ").upper()
                if done == "Y":
                    return dOrd
                else:
                    continue
            else:
                print("Wrong input. Try again.")       
    # Method that saves different stages of the preprocessing algorithm
    def StagesOutput(self, d, images, count, folder):
        i = 0                       # Start a counter
        for key,value in d.items(): # Loop through the dictionary
            if value[0] == True:    # If the user wants to extract a step of preprocessing
                cv2.imwrite("./{4}/{3}frame{0}.{1}{2}.bmp".format(count,value[1],key,self.name,folder), images[i]) # Save images with proper name format
                size = os.path.getsize("./{4}/{3}frame{0}.{1}{2}.bmp".format(count,value[1],key,self.name,folder)) # Get the size of the output
            i += 1                  # Increase counter
        return size 
    # Method that uses indexes returned by the smart cropping to crop frames 
    def CropImages(self, img, start_row,end_row, start_col,end_col):
        height, width = img.shape[:2]                         # Get the height and width of the images
        cropped = img[start_row:end_row , start_col:end_col]  # Use indexing to crop out the desired rectangle
        return cropped
    # Method that return indexes to crop images based on the location of rivets and also returns slope of the rivet line and the target image for damage detection
    def CropImagesSmart(self, img, showCircles = False):      # Method coded with assistance of Laith Sakka
        img = cv2.GaussianBlur(img,(5,5),0)                   # Blur the image to avoid picking incorrect circles and help LAMINART)
        target = copy.deepcopy(img)                           # Create a copy of the image to create a target for error detection
        rivets = copy.deepcopy(img)
        (thresh, target) = cv2.threshold(target, 0, 255, cv2.THRESH_BINARY)    # Set the target image as a solid white background where re ideal rivets will be draw
        try:
            circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,100,param1=20,param2=40,minRadius=5,maxRadius=50)   # Detect circles using Hough transform
            circles = np.uint16(np.around(circles))           # Transform circles into a numpy array
        except AttributeError :                               # Let the user know there was an error in detecting the circles   
            print(" Error detecting circles")
            return
        maxH = int(0)   # Create variables for the maximum and minimum lenghts and heights
        minH = int(1000000)
        maxL = int(0)
        minL = int(100000)
        maxR = 0.0 
        radii = [item[2] for item in circles[0]]  # Get all of the radii of the circle
        medR = int(np.median(radii))              # Determine the median radius 
        for i in circles[0, :]:
            r = i[2]
            maxR = max(maxR, r)
            maxL = max(maxL, i[0]+r)
            minL = min(minL, i[0]-r)
            maxH = max(maxH, i[1]+r)
            minH = min(minH, i[1]-r) 
            cv2.circle(target, (i[0], i[1]), medR, (0, 255, 0), 6)     # Draw the outer circles in the target image 
        Sorted = circles[0][circles[0][:, 0].argsort()].tolist()       # Rotate the picture (1) sort circles based on x axis
        mid = int(len(Sorted)/2)-1                                     # Fin middle element in sorted circles
        MidPoint = Sorted[mid][0]                                      # Find first column of circles 
        Group1 = [i for i in Sorted if abs(i[0]-MidPoint) < maxR]
        xs = [i[0] for i in Group1]
        ys = [i[1] for i in Group1]   
        # Compute the slope of the line connecting group one
        slope = linregress(xs, ys)[0]                          # Slope in units of y / x
        slope_angle = math.atan(slope)                         # Slope angle in radians
        slope_angle_degrees = -abs(math.degrees(slope_angle))  # Slope angle in degrees   
        slope_angle_degrees += 180
        slope_angle_degrees -= 45
        indexes = [minH-15,maxH+15, minL-15,maxL+15] 
        return img,indexes,slope_angle_degrees,target,rivets
    # Method that gets user input to create a folder where the output is going to be saved
    def SaveFolder(self):
        directory = input("Enter name of the directory to save images for video {0}: ".format(self.name)) # Get the name of the folder
        if directory =="":
            self.SaveFolder()
        try:    #Handle errors in creating the folder
            if not os.path.exists('./{0}/'.format(directory)): # If the folder does not exist 
                os.makedirs(directory)                         # Create the folder 
                return True, directory                         # Return true for the flag and the name of the folder
            else:                                              # Otherwise, if the folder already exists 
                 # Ask the user if he wants to save to the existing folder
                 save = input(("{0} directory already exists. Do you want to save to that folder? Press Y for yes, any other for no. ".format(directory))).upper() 
                 if save != "Y": # If he does not want to save to the existing folder run the method again to allow for a new folder to be created
                     return self.SaveFolder()
                 else:                                         # Save to the already existing folder
                    return True, directory                     # Return true for the flag and the name of the folder
        except OSError:                                        # If an exception is raised 
            print ('Error: Creating directory. ' +  directory) # Let the user know there was an error
            return False,None # Return false for the flag and none for the name of the folder
    # Method to resize images conserving aspect ratio
    def ImageResize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        dim = None
        try:                                        # Initialize the dimensions of the image to be resized and grab the image size
            (h, w) = image.shape[:2]
            if width is None and height is None:    # If both the width and height are None, then return the original image
                return image
            try:
                if width is None:                   # Check to see if the width is None
                    r = height / float(h)           # Calculate the ratio of the height and construct the dimensions
                    dim = (int(w * r), height)  
                else:                               # Otherwise, the height is None
                        r = width / float(w)        # Calculate the ratio of the width and construct the dimensions
                        dim = (width, int(h * r))
            except TypeError:
                   return self.image_resize()     
        except AttributeError:
            return image
        resized = cv2.resize(image, dim, interpolation = inter)     # Resize the image
        return resized                                              # Return the resized image
    # Method to increase brightness and contrast
    def BrightnessAndContrast(self, img): 
        if self.histCon:
            transformed = cv2.equalizeHist(img)
        else:
            transformed = cv2.addWeighted(img,self.alpha, np.zeros(img.shape, img.dtype), 0,self.beta)
        return transformed
    # Method to rotate frames with a white background
    def RotateBound(self,image, angle):
        (h, w) = image.shape[:2]                               # Grab the dimensions of the image and then determine the center
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)     # Grab the rotation matrix (applying the negative of the
        cos = np.abs(M[0, 0])                                  # angle to rotate clockwise), then grab the sine and cosine
        sin = np.abs(M[0, 1])                                  # (i.e., the rotation components of the matrix)
        nW = int((h * sin) + (w * cos))                        # Compute the new bounding dimensions of the image
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX                               # Adjust the rotation matrix to take into account translation
        M[1, 2] += (nH / 2) - cY
        # Perform the actual rotation with a white background and return the image
        return cv2.warpAffine(image, M, (nW, nH),borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    # Method to create a text file containing the angles used for rotations
    def CreateTextFile(self,rotations,folder):
        f= open("./{0}/rotations.txt".format(folder),"w+")
        for rotation in rotations:
            f.write(str(rotation)+",")
        f.close()
    # Method that extract frames 
    def ExtractFrames(self):
        saveFlag, directory = self.SaveFolder()                      # Call the method to create a folder to store the output
        if saveFlag == False:                                        # If there was a problem creating the folder exit the method
            return                                                   # Exit the method if there was an error in creating the folder
        cap = cv2.VideoCapture(self.name+".MP4")                     # Get the video 
        while not cap.isOpened():                                    # Wait for the video to open
            cap = cv2.VideoCapture(self.name+".MP4")                 # Try to get the video again
            print ("Waiting for video to open")                      # Let the user know that the video is being opened 
            time.sleep(0.01)                                         # Add a small delay to wait for the video to open
        self.fps = cap.get(cv2.CAP_PROP_FPS)                         # Read the frames per second of the video
        self.frameStep = round(self.timeStep * self.fps)             # Get a rounded frame step to extract frames
        print("Frame step: " + str(self.frameStep))                  # Print number of frames per second of the video
        count = 0                                                    # Start counter of the frames
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))              # Get total number of frames in the video
        print("Number of frames: " + str(length))                    # Print the number of frames in the video
        images = []                                                  # Create a list to store different stages of the algorithm
        time.sleep(0.01)                                             # Wait a small time before getting input to avoid errors
        d = self.GetInput()                                          # Get parameters from the user   
        avgSize = 0                                                  # Create a variable to hold the average size of the output      
        rotations = []                                               # Keep track of rotation angles
        gaussian = False                                             # Determine if gaussian adaptive threshold will be used
        shadow = False#not gaussian                                        # If gaussian adaptive threshold is not used shadow must be removed
        start_time = time.time()                                     # Start timer to check duration of computation
        while True:                                                  # Main loop of the program                                      
            flag, frame = cap.read()                                 # Read a frame
            if flag :                                                # If reading succeeded                                          
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)-1       # Get the frame number and substract 1 so the idexing starts at 0
                print ("Frame: "+str(pos_frame), end="", flush=True) # Print the frame number
                if count%self.frameStep==0:                          # If the counter is a multiple of the desired frame step to extract an image    
                    print(" Extracting otput from this frame.",end="", flush=True)
                    images.append(frame)                             # Add the original frame to images list
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Transform the frame into grayscale             
                    images.append(img)                               # Add the grayscale frame to images list
                    try:
                        img,indexes,rotate_angle,target,rivets = self.CropImagesSmart(img)       # Obtain indexes to use cropping and the angle of the rotation
                    except TypeError:                                          # If there is an error in calling CropImagesSmart
                        print("Error extracting rotation angle.")
                        h,w = frame.shape[:2]                                  # Get height and width of the original frame
                        indexes = [0,h,0,w]                                    # Use these indexes to get the whole frame
                        rotate_angle = 0                                       # There's no need to rotate as this frame has failed
                        target = img                                           # Do nothing 
                        rivets = img                                           # Do nothing 
                    rotations.append(rotate_angle)                             # Save the angle needed for rotation
                    target = self.CropImages(target, indexes[0],indexes[1],indexes[2],indexes[3])   # Crop to get only the portion of the frame with the rivets
                    imgc = self.CropImages(img, indexes[0],indexes[1],indexes[2],indexes[3])        # Crop to get only the portion of the frame with the rivets
                    rivets = self.CropImages(rivets, indexes[0],indexes[1],indexes[2],indexes[3])   # Crop to get only the portion of the frame with the rivets
                    images.append(imgc)                                        # Add the cropped frame to images list
                    ig = self.BrightnessAndContrast(imgc)                      # Use histogram equalization to increase contrast (or simple brightness and contrast modification)                        
                    images.append(ig)                                          # Add the contrast enhanced frame to images list
                    rivets = self.BrightnessAndContrast(rivets)                # Use histogram equalization to increase contrast (or simple brightness and contrast modification)                        
                    if gaussian:                                               # Use gaussian adaptive thershold or not
                        ibw = cv2.adaptiveThreshold(ig,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                        rivets = cv2.adaptiveThreshold(rivets,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                    else:
                        (thresh, ibw) = cv2.threshold(ig, 145, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)        # Convert image to black and white with threshold method             
                        (thresh, rivets) = cv2.threshold(rivets, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU) # Convert image to black and white with threshold method             
                    if shadow:                                                 # Eliminate are with shadow
                        height = int(ig.shape[0]*0.7)                              
                        width = int(ig.shape[1])
                        ibw = self.CropImages(ibw, 0,height,0,width)
                        rivets = self.CropImages(rivets, 0,height,0,width)
                        target = self.CropImages(target, 0,height,0,width)
                    images.append(ibw)                                         # Add the black and white frame to images list    
                    ibws = self.ImageResize(ibw, height = self.sizey)          # Resize the frame using area interpolation       
                    images.append(ibws)                                        # Add the resized frame to images list
                    try:
                        rotated = self.RotateBound(ibws, angle=rotate_angle)
                    except (AttributeError,ValueError):
                        print("Error")
                        rotated = ibw
                    images.append(rotated)
                    images.append(target)
                    images.append(rivets)
                    size = self.StagesOutput(d, images, count, directory)      # Save frames according to user input
                    avgSize += size                                            # Increase the size with the size of the current output
                    images = []                                                # Empty images list for next iteration     
                print("")                                                      # Print a new line
                count+=1                                                       # Increase counter
                workingTime = time.time()
            else:                                                              # If reading the frame failed                                             
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)-1
                print ("Frame "+str(pos_frame)+" is not ready")                # The next frame is not ready, so try to read it again
                time.sleep(0.01)                                               # It is better to wait for a while for the next frame to be ready
                notReadyTime = time.time()- workingTime
                if notReadyTime > 20:
                    print("Computation took too long to run, there must be something wrong with the video file")
                    break
            if cv2.waitKey(10) == 27:                                          # If the esc key is pressed break out of the loop
                avgSize /= 1024*(count/self.frameStep)                         # Calculate the average size of the output
                print("The average size of the output is {0:.2f} KB.".format(avgSize))  # Let the user know the average size of the output
                break                                                                   # Break out of the main loop
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT):   # If the number of captured frames is equal to the total number of frames stop
                avgSize /= 1024*(count/self.frameStep)                                  # Calculate the average size of the output
                print("The average size of the output is {0:.2f} KB.".format(avgSize))  #Let the user know the average size of the output
                break                                                                   # Break out of the main loop
        self.CreateTextFile(rotations,directory)
        cap.release()                                                                   # Release the video
        print("Duration of computation: %.2f seconds." % (time.time() - start_time))    # Print how long it took the program to run       
    # Method included for rapid testing
    def TestProcessing(self,name):                      # Test preprocessing on an unaltered frame
        img = cv2.imread(name+".bmp")                   # Read the frame to be processed 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Transform the frame into grayscale      
        img = cv2.GaussianBlur(img,(5,5),0)             # Blur the image to help in avoid picking incorrect circles
        cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
        width = int(img.shape[0]/2)
        height =  int(img.shape[1]/2)
        cv2.resizeWindow('Image', height,width)         # Resize the window so that the image is completely shown on the screen
        cv2.imshow('Image',img)                         # Display the image to see the result
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,100,param1=20,param2=40,minRadius=5,maxRadius=50) # Detect the circles in the image
        print("The found circles are: ")
        print(circles) 
        try: 
            circles = np.uint16(np.around(circles))
            print("The circles converted to an numpy array are: ")
            print(circles)
            maxH = int(0)
            minH = int(1000000)
            maxL = int(0)
            minL = int(100000)
            maxR = 0.0
            radii = [item[2] for item in circles[0]]  # Get all of the radii of the circle
            medR = int(np.median(radii))              # Determine the median radius 
            target = copy.deepcopy(img)    # Create a copy of the image to show where the dectected circles are
            (thresh, target) = cv2.threshold(target, 0, 255, cv2.THRESH_BINARY)
            for i in circles[0, 0:]:
                r = i[2]    # Print(i)
                maxR = max(maxR, int(r))
                maxL = max(maxL, int(i[0]+r))
                minL = min(minL, int(i[0]-r))
                maxH = max(maxH, int(i[1]+r))
                minH = min(minH, int(i[1]-r))
                cv2.circle(target, (i[0], i[1]), medR, (0, 0, 0), 6)       # Draw the outer circle 
            print("The maximum and minumum values are: ")
            print(maxR,maxL,minL,maxH,minH)
            cv2.namedWindow('Target image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Target image', height,width) 
            cv2.imshow('Target image',target)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            indexes = [minH,maxH+15, minL-15,maxL+15]
            img = self.CropImages(img, indexes[0],indexes[1],indexes[2],indexes[3])
            width = int(img.shape[0]/2)
            height =  int(img.shape[1]/2)
            cv2.namedWindow('Cropped image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Cropped image', height,width)
            cv2.imshow('Cropped image',img)
            cv2.waitKey(0)
            target = self.CropImages(target, indexes[0],indexes[1],indexes[2],indexes[3])
            cv2.imwrite("drawn circles.bmp", target)
            cv2.namedWindow('Image with circles',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image with circles', height,width) 
            cv2.imshow('Image with circles',target)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("No drawn circles.bmp", img)
            # now we want to rotate the picture
            # (1) sort circles based on x axis
            Sorted = circles[0][circles[0][:, 0].argsort()].tolist()
            print("Sorted elements are: ")
            print(Sorted)
            # find first column of circles :
            mid = int(len(Sorted)/2)-1
            MidPointX = Sorted[mid][0]
            Group1 = [i for i in Sorted if i[0]-MidPointX > maxR]
            print(Group1)
            xs = [i[0] for i in Group1]
            xs = xs[1:int(len(xs)/2)+2]
            ys = [i[1] for i in Group1]
            ys = ys[1:int(len(ys)/2)+2]
            print(xs)
            print(ys)
            # compute the slope of the line connecting group one\
            slope = linregress(xs, ys)[0]  # slope in units of y / x
            slope_angle = math.atan(slope)  # slope angle in radians
            slope_angle_degrees = -abs(math.degrees(slope_angle)) # slope angle in degrees
            print("Slope: " + str(slope_angle_degrees))
            slope_angle_degrees += 180
            slope_angle_degrees -= 45
            img = self.BrightnessAndContrast(img)
            cv2.namedWindow('Image with enhanced contrast',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image with enhanced contrast', height,width) 
            cv2.imshow('Image with enhanced contrast',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            cv2.namedWindow('Image black and white',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image black and white', height,width) 
            cv2.imshow('Image black and white',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img = self.ImageResize(img, height = 256)
            target = self.ImageResize(target, height = 256)
            cv2.imshow('Image resized',img)
            cv2.imshow('Image resized2',target)
            cv2.imwrite("damaged1.bmp", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            ig = self.RotateBound(img, angle=slope_angle_degrees)
            igc = self.RotateBound(target, angle=slope_angle_degrees)
            cv2.imshow('Image rotated',ig)
            cv2.imshow('Image rotated2',igc)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("damaged2.bmp", ig)
            cv2.imwrite("target.bmp", igc)
        except AttributeError: 
            print("Error")
# Define a function to handle errors, create preprocessing object and extract the frames 
def RunPreprocessing(): # Videos need to be in the same folder as the program
   choice = ""
   while choice != ("P" or "T"):
        choice = input("Press P to run main pre-processing code or T to run testing code. ").upper()
        if choice == "P":
            videos = input("Enter the name of the videos that you want to pre-process separeted by a comma. ").split(",") # Ask the user for the names of the videos   
            if videos[0] =="":                                  # If the user did not entered anything
                RunPreprocessing()                              # Rerun the function
            else:
                p = os.getcwd()                                 # Get the path of the program
                for vid in videos:                              # Run a loop through the videos that the user provided
                    f = Path("{0}/{1}.MP4/".format(p,vid))      # Get the video file
                    if not f.is_file():                         # Check to see if there is a file with the name provided by the user
                         run = input("{0} is a wrong name of file. Press Y to run the program again, any other key to exit. ".format(vid)).upper() # Let the user know there was an error
                         videos.remove(vid)                     # Get rid of the wrong input
                         if run != "Y":                         # If the user wants to exit
                            return                              # Exit the function
                         else:                                  # If the user wants to provide a new name run the function again
                            RunPreprocessing()
                    v= Preprocessing(vid)                       # Create an object of class Preprocessing
                    v.ExtractFrames()                           # Call the extractFrames method to get output from the video  
        elif choice =="T":
            name = input("Enter the name of the frame you want to test. ")
            try:
                v= Preprocessing(name)
                v.TestProcessing(v.name)
            except:
                print("Wrong name of file. Try again. ")
                RunPreprocessing()
        else:
            return
# Run the function
RunPreprocessing()