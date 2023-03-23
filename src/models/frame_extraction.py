import cv2
import os

config={
    'root':{'dir':'/home/sdastani/scratch/ucf101/small_ucf101'}
}

# for folder in os.listdir(config['root']['dir']):
os.chdir('../datasets/frames')
for video in os.listdir(config['root']['dir']):
    cam = cv2.VideoCapture(os.path.join(config['root']['dir'], video))
    try:
        if not os.path.exists(video):
            os.makedirs(video)
    except OSError:
        print ('Error: Creating directory of data')
    currentframe = 0
    while(True):
        ret,frame = cam.read()
        if ret:
            name = f'./{video}/frame' + str(currentframe) + '.jpg'
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    print(f'video {video} finished')