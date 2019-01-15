import os
import shutil

# copies all .JPG files from a given array of folders into a target folder

# put PATHs to directories in the array
RootDir1 = [r'INSERT_DIRECTORY_PATH_HERE', r'INSERT_DIRECTORY_PATH_HERE']

# PATH to target directory
TargetFolder = r'INSERT_DIRECTORY_PATH_HERE'

for paths in RootDir1:
    for root, dirs, files in os.walk((os.path.normpath(paths)), topdown=False):
        for name in files:
            if name.endswith('.JPG'):
                print("Found: ", name)
                SourceFolder = os.path.join(root, name)
                shutil.copy2(SourceFolder, TargetFolder)