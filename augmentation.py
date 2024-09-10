import os
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch

# Define the paths
input_images_path = 'dataset/images'
input_masks_path = 'dataset/masks'
output_images_path = 'dataset_U2/images'
output_masks_path = 'dataset_U2/masks'


# Create the output folder if it does not exist
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_masks_path, exist_ok=True)

def augmentation(input_path, filename, output_path):
    # Open the image file
    image_path = os.path.join(input_path, filename)
    image = Image.open(image_path)
    ### Augmentation
    croped_img = v2.RandomCrop(size=(224, 224))(image)
    blured_img = v2.GaussianBlur(kernel_size=9, sigma=(0.8, 1.2))(image)
    brightnessed_img = v2.ColorJitter(brightness=(1.1,1.5))(image)
    darknessed_img = v2.ColorJitter(brightness=(0.6,0.9))(image)

    # Save the augmented images (if needed)
    split_filename = filename.split('.')
    name = split_filename[0]
    surname = split_filename[1]
    croped_img.save(os.path.join(output_path, f'{name}_cropped.{surname}'))
    blured_img.save(os.path.join(output_path, f'{name}_blurred.{surname}'))
    brightnessed_img.save(os.path.join(output_path, f'{name}_brightnessed.{surname}'))
    darknessed_img.save(os.path.join(output_path, f'{name}_darknessed.{surname}'))


# Loop through each file in the input folder
for input_path in [input_images_path, input_masks_path]:
    for filename in os.listdir(input_path):
        if filename.endswith('.jpg'):
            output_path = output_images_path
            augmentation(input_path, filename, output_path)

        elif filename.endswith('.png'):
            output_path = output_masks_path
            augmentation(input_path, filename, output_path)
