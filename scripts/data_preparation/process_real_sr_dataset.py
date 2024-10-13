import os 
import shutil
from tqdm import tqdm
import argparse

PATH = r"C:\Users\abreo\Downloads\archive (2)\RealSR (ICCV2019)"
MAIN_DIRECTORY = "merged"
DIR_PATH = os.path.join(PATH ,  MAIN_DIRECTORY)


def create_directory_structure(sub_sub_directory):

    subdirectories = ["Train", "Val"]
    # subsubdirectories = ["HR","X2", "X3", "X4"]
    subsubdirectories = ["HR" , sub_sub_directory]

    # Create the main directory
    os.makedirs( DIR_PATH ,exist_ok=True)  # Create main directory

    # Create subdirectories and their sub-subdirectories
    for subdir in subdirectories:
        # Path for the current subdirectory
        subdir_path = os.path.join(DIR_PATH, subdir)
        
        # Create the subdirectory
        os.makedirs(subdir_path, exist_ok=True)
        
        # Create sub-subdirectories inside the current subdirectory
        for subsubdir in subsubdirectories:
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            os.makedirs(subsubdir_path, exist_ok=True)
            print(f"Created directory {subsubdir_path}")


def transfer_data(type , scale_directory):

    camera_lens = ["Canon" , "Nikon"]
    for lens in camera_lens:
        if type == "Val":
            base_path = os.path.join(PATH , f"{lens}\Test")
        else:
            base_path = os.path.join(PATH , f"{lens}\{type}")
        
        for root , dirs , files in os.walk(base_path):
            print(f"Current Directory: {root}")
            
            for dir_name in dirs:
                if dir_name == scale_directory:
                    print(f"  Subdirectory: {dir_name}")
                    
                    subdir_path = os.path.join(root, dir_name)
                    
                    subdir_files = os.listdir(subdir_path)
                    for file in tqdm(subdir_files):
                        file_path = os.path.join(subdir_path, file)
                        if 'HR' in file:
                            try:
                                new_file_path = file_path[:-7] + ".png"
                                os.rename(file_path, new_file_path)
                            except FileNotFoundError:
                                print(f"Error: The file '{file_path}' does not exist.")
                            except FileExistsError:
                                print(f"Error: A file named '{file_path[:-3]}' already exists in the directory.")
                            except Exception as e:
                                print(f"An error occurred: {e}")

                            destination_path = os.path.join(DIR_PATH , f'{type}\HR')
                            shutil.copy(new_file_path, destination_path)
                        elif 'LR2' in file:
                            destination_path = os.path.join(DIR_PATH , f'{type}\X2')
                            shutil.copy(file_path, destination_path)
                        elif 'LR3' in file:
                            destination_path = os.path.join(DIR_PATH , f'{type}\X3')
                            shutil.copy(file_path, destination_path)
                        elif 'LR4' in file:
                            destination_path = os.path.join(DIR_PATH , f'{type}\X4')
                            shutil.copy(file_path, destination_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str, required=True)
    parser.add_argument('--new_directory_name', type=str, required=True)
    parser.add_argument('--scale', type=str, required=True)

    args = parser.parse_args()

    PATH = args.directory_path
    MAIN_DIRECTORY = args.new_directory_name
    DIR_PATH = os.path.join(PATH ,  MAIN_DIRECTORY)


    create_directory_structure("X" + args.scale)
    transfer_data('Train' , args.scale)
    transfer_data('Val' , args.scale)


# python process_real_sr_dataset.py --directory_path "C:\\Users\\abreo\\Downloads\\archive (2)\\RealSR (ICCV2019)" --new_directory_name "merged"  --scale "2"
                            






