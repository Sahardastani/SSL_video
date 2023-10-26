import os
import csv

# Specify the directory containing your .mp4 videos
videos_directory = '/home/as89480@ens.ad.etsmtl.ca/SSL_video/ucf101/UCF101'

# Create a list to store the directories
directories = []

# Iterate through the files in the directory
for root, dirs, files in os.walk(videos_directory):
    for file in files:
        if file.endswith(".avi"):
            # Add the directory path to the list
            directories.append(os.path.join(root, file))

# Specify the path for the CSV file
csv_file = 'directories.csv'

# Write the directories to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    for directory in directories:
        writer.writerow([directory])

print(f"Directories saved to {csv_file}")
