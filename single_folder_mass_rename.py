import os
import shutil

# This program renames all .JPG files in given folders based on their classification label

# Array of paths to directories to rename
RootDir1 = [
r'INSERT_DIRECTORY_PATH_HERE'
]

# numbers added to end of each file for uniqueness
af = 0
an = 0
di = 0
ha = 0
ne = 0
sa = 0
su = 0


for paths in RootDir1:
    for root, dirs, files in os.walk((os.path.normpath(paths)), topdown=False):
        for name in files:
            if name.endswith('.JPG'):
                print("Found: ", name)
                if (name[4:6] == 'AF'):
                    new_name = "afraid" + str(af) + ".JPG"
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                    af = af + 1
                elif (name[4:6] == 'AN'):
                    new_name = "angry" + str(an) + ".JPG"
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                    an = an + 1
                elif (name[4:6] == 'DI'):
                    new_name = "disgusted" + str(di) + ".JPG"
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                    di = di + 1
                elif (name[4:6] == 'HA'):
                    new_name = "happy" + str(ha) + ".JPG"
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                    ha = ha + 1
                elif (name[4:6] == 'NE'):
                    new_name = "neutral" + str(ne) + ".JPG"
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                    ne = ne + 1
                elif (name[4:6] == 'SA'):
                    new_name = "sad" + str(sa) + ".JPG"
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                    sa = sa + 1
                elif (name[4:6] == 'SU'):
                    new_name = "suprised" + str(su) + ".JPG"
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                    su = su + 1
                else:
                    print("error")