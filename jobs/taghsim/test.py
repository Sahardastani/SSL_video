import os
import zipfile

def zip_directory(directory_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for foldername, _, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))

# Specify the directory path and output zip file path
directory_path = '/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/taghsim/frames'
output_path = '/home/sdastani/projects/rrg-ebrahimi/sdastani/SSL_video/taghsim/g.zip'

# Call the function to zip the directory
zip_directory(directory_path, output_path)





# import os
# import random
# import shutil

# # Set the source directory containing the 300 videos
# source_directory = "/home/sdastani/scratch/datasets/AVA/videos_15min"

# # Create the six target directories
# target_directories = ["a", "b", "c", "d", "e",
#                         "f", "g", "h", "i", "j", 
#                         "k", "l", "m", "n", "o"]
# for directory in target_directories:
#     os.makedirs(directory, exist_ok=True)

# # Get a list of all the video files in the source directory
# video_files = [file for file in os.listdir(source_directory)]

# # Shuffle the list of video files
# random.shuffle(video_files)

# # Distribute the videos into the target directories
# for i, video in enumerate(video_files):
#     target_directory = target_directories[i % 15]  # Distribute videos evenly among the six directories
#     source_path = os.path.join(source_directory, video)
#     target_path = os.path.join(target_directory, video)
#     shutil.copy(source_path, target_path)
