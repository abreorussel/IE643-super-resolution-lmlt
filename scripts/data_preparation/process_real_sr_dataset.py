import os 
import shutil
from tqdm import tqdm
import argparse
import cv2

def downsample_image( directory_path , scale_factor):
    print("Downsampling the images ..................")
    scale_path = os.path.join(directory_path, 'X'+args.scale)
    for file in tqdm(os.listdir(scale_path)):
        input_path = os.path.join(scale_path , file)
        img = cv2.imread(input_path)
        h, w = img.shape[:2]
        new_size = (w // scale_factor, h // scale_factor)
        downsampled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(input_path, downsampled_img)


def create_directory_structure(sub_sub_directory):

    subdirectories = ["Train", "Val"]
    subsubdirectories = ["HR" , sub_sub_directory]

    # Create the main directory
    os.makedirs( DIR_PATH ,exist_ok=True)  

    # Create subdirectories and their sub-subdirectories
    for subdir in subdirectories:

        subdir_path = os.path.join(DIR_PATH, subdir)
        
        # Create the subdirectory
        os.makedirs(subdir_path, exist_ok=True)
        
        # Create sub-subdirectories inside the current subdirectory
        for subsubdir in subsubdirectories:
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            os.makedirs(subsubdir_path, exist_ok=True)
            print(f"Created directory {subsubdir_path}")


def transfer_data(type , scale_directory):
    print("Transferring Files ........................")
    camera_lens = ["Canon", "Nikon"]
    hr_counter = 1  # Counter for HR images
    lr2_counter = 1  # Counter for LR2 images
    lr3_counter = 1  # Counter for LR3 images
    lr4_counter = 1  # Counter for LR4 images
    for lens in camera_lens:
        if type == "Val":
            base_path = os.path.join(PATH , f"{lens}/Test")
        else:
            base_path = os.path.join(PATH , f"{lens}/{type}")

        # print(f"Base Path: {base_path}")
        
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
                                new_file_name = f"{hr_counter:03d}.png"
                                new_file_path = os.path.join(subdir_path, new_file_name)
                                os.rename(file_path, new_file_path)
                            except FileNotFoundError:
                                print(f"Error: The file '{file_path}' does not exist.")
                            except FileExistsError:
                                print(f"Error: A file named '{file_path[:-3]}' already exists in the directory.")
                            except Exception as e:
                                print(f"An error occurred: {e}")

                            destination_path = os.path.join(DIR_PATH , f'{type}/HR')
                            shutil.copy(new_file_path, destination_path)

                            hr_counter += 1

                        elif 'LR2' in file:
                            try:
                                new_file_name = f"{lr2_counter:03d}x2.png"
                                new_file_path = os.path.join(subdir_path, new_file_name)
                                os.rename(file_path, new_file_path)
                            except FileNotFoundError:
                                print(f"Error: The file '{file_path}' does not exist.")
                            except FileExistsError:
                                print(f"Error: A file named '{file_path[:-3]}' already exists in the directory.")
                            except Exception as e:
                                print(f"An error occurred: {e}")
                            destination_path = os.path.join(DIR_PATH , f'{type}/X2')
                            shutil.copy(new_file_path, destination_path)

                            lr2_counter += 1
                            
                        elif 'LR3' in file:
                            try:
                                new_file_name = f"{lr3_counter:03d}x3.png"
                                new_file_path = os.path.join(subdir_path, new_file_name)
                                os.rename(file_path, new_file_path)
                            except FileNotFoundError:
                                print(f"Error: The file '{file_path}' does not exist.")
                            except FileExistsError:
                                print(f"Error: A file named '{file_path[:-3]}' already exists in the directory.")
                            except Exception as e:
                                print(f"An error occurred: {e}")
                            destination_path = os.path.join(DIR_PATH , f'{type}/X3')
                            shutil.copy(new_file_path, destination_path)

                            lr3_counter += 1

                        elif 'LR4' in file:
                            try:
                                new_file_name = f"{lr4_counter:03d}x4.png"
                                new_file_path = os.path.join(subdir_path, new_file_name)
                                os.rename(file_path, new_file_path)
                            except FileNotFoundError:
                                print(f"Error: The file '{file_path}' does not exist.")
                            except FileExistsError:
                                print(f"Error: A file named '{file_path[:-3]}' already exists in the directory.")
                            except Exception as e:
                                print(f"An error occurred: {e}")
                            destination_path = os.path.join(DIR_PATH , f'{type}/X4')
                            shutil.copy(new_file_path, destination_path)

                            lr4_counter += 1


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
    downsample_image(os.path.join(DIR_PATH , 'Train'), int(args.scale))
    transfer_data('Val' , args.scale)
    downsample_image(os.path.join(DIR_PATH , 'Val'), int(args.scale))

    shutil.rmtree(os.path.join(PATH , "Canon"))
    shutil.rmtree(os.path.join(PATH , "Nikon"))



# Script to execute the preprocessing step
# python process_real_sr_dataset.py --directory_path "C:\\Users\\abreo\\Downloads\\archive (2)\\RealSR (ICCV2019)" --new_directory_name "realsr"  --scale "4"
                            






