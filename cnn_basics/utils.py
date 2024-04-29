import os


# Check if the folder already exists, if not, create it
def create_cnn_data_folder():
    current_directory = os.getcwd()
    new_folder = "cnn_data"
    new_folder_path = os.path.join(current_directory, new_folder)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return f"{current_directory}\{new_folder}"
