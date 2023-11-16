def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [line.strip() for line in lines]

# Example usage
file_path = '/home/lintzuh@kean.edu/BUS/AMR/record/testSetWithoutNormal.txt'  # Replace with the path to your text file
file_content = read_file_to_list(file_path)
print(file_content[:5])
