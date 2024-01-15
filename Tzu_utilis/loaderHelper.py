def getType(fileName):
    if fileName[0] == 'b':
        type = 'benign'
    elif fileName[0] == 'm':
        type = 'malignant'
    else:
        type = 'normal'

    return type


def readFromList(file_path, root):
    import os
    formatted_list = []
    label_list=[]

    # Open the file for reading
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line

        # Iterate over each line in the file
        for line in file:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
            
            type = getType(line)
            if type == 'normal':
                label = 0
            else:
                label = 1
        
            picPath = os.path.join(root, type, line+'.png')
            # Format the line and add it to the list
            formatted_list.append(picPath)
            label_list.append(label)

    return formatted_list, label_list
def readFromListWithoutNormal(file_path, root):
    import os
    formatted_list = []
    label_list=[]

    # Open the file for reading
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line

        # Iterate over each line in the file
        for line in file:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
            
            type = getType(line)
            if type == 'normal':
                continue
            else:
                label = 1
                picPath = os.path.join(root, type, line+'.png')
                # Format the line and add it to the list
                formatted_list.append(picPath)
                label_list.append(label)

    return formatted_list, label_list


def readFromListWithoutNormalAndShortName(file_path, root):
    import os
    formatted_list = []
    label_list=[]

    # Open the file for reading
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line

        # Iterate over each line in the file
        for line in file:
            # Strip whitespace and skip empty lines
            line = line.strip()
            if not line:
                continue
            
            type = getType(line)
            if type == 'normal':
                continue
            else:
                label = 1
                picPath = (line+'.png')
                # Format the line and add it to the list
                formatted_list.append(picPath)
                label_list.append(label)

    return formatted_list, label_list