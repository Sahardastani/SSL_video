import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', 
                    type=str, 
                    default='/home/sdastani/scratch/ucf101/small_ucf101', 
                    help='The directory of UCF101 videos')
                    
args = parser.parse_args()

if not os.path.exists('frames'):
    os.makedirs('frames')

for video in os.listdir(args.root):
    cam = cv2.VideoCapture(os.path.join(args.root, video))
    try:
        if not os.path.exists(f'./frames/{video}'):
            os.makedirs(f'./frames/{video}')
    except OSError:
        print ('Error: Creating directory of data')
    currentframe = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            name = f'./frames/{video}/frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    cam.release()
    # cv2.destroyAllWindows()
    print(f'video {video} finished')