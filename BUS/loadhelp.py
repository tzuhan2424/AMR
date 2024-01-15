def getPathShortName(fullpath):
    import os
    directory, img_name = os.path.split(fullpath)
    subdirectories = directory.split('/')
    type = subdirectories[-1]

    return img_name, type