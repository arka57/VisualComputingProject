import os
import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision.datasets import ImageFolder
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import threading  
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="accelerate")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Load the words.txt file and extract the labels
with open('/home/rs1/24CS91R03/dataset/words.txt', 'r') as f:
    label_map = {line.strip().split('\t')[0]: line.strip().split('\t')[1].split(',')[-1].strip() for line in f}


# Set up the BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to('cuda:0')

# Load the base tiny-imagenet dataset
base_dir = '/home/rs1/24CS91R03/dataset/train'
dataset = ImageFolder(base_dir)

# Create a DataFrame with image paths and labels
df = pd.DataFrame({
    "image_path": [os.path.join(base_dir, dataset.imgs[i][0]) for i in range(len(dataset))],
    "label_no": [dataset.imgs[i][1] for i in range(len(dataset))]
})
df['label'] = df['label_no'].apply(lambda x: label_map[dataset.classes[x]])

# Generate captions using BLIP model
def generate_caption(image_path, label):
    image = Image.open(image_path).convert('RGB')
    prompt = f"{label}"   #pass the image label as prompt
    inputs = processor(image, prompt, return_tensors="pt").to('cuda:0')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
        return processor.decode(out[0], skip_special_tokens=True)

df['caption'] = df.apply(lambda row: generate_caption(row['image_path'], row['label']), axis=1)
print("All Image Captions Generated")

# Split DataFrame into two halves for parallel processing
df_split1 = df.iloc[:len(df) // 2].copy()
df_split2 = df.iloc[len(df) // 2:].copy()

# Initialize Stable Diffusion pipelines for two GPUs
pipe1 = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe1 = pipe1.to("cuda:0")
pipe1.safety_checker = None 

pipe2 = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe2 = pipe2.to("cuda:1")
pipe2.safety_checker = None
 
# Define a function to process a DataFrame chunk on a given GPU
def process_images(df_chunk, pipe, gpu_id):
    for idx, row in tqdm(df_chunk.iterrows(),total=len(df_chunk)):
        caption = row['caption']
        original_image_path = row['image_path']
        folder_path = os.path.dirname(original_image_path)

        generated_image_paths = []
        for i in range(4):
            image = pipe(
                caption,
                negative_prompt="Low quality, Low res, Blurry, Jpeg artifacts, Grainy, Cropped, Out of frame, Out of focus, Bad anatomy, Bad proportions, Deformed, bad photo, aberrations, creative, drawing, unnatural",
                num_inference_steps=20,
                guidance_scale=7.0
            ).images[0]

            generated_image_path = os.path.join(folder_path, f'{idx}_{i}.png')
            image.save(generated_image_path)
            generated_image_paths.append(generated_image_path)

        df_chunk.at[idx, 'generated_image_paths'] = str(generated_image_paths)

# Create threads for parallel processing on two GPUs
thread1 = threading.Thread(target=process_images, args=(df_split1, pipe1, 0))
thread2 = threading.Thread(target=process_images, args=(df_split2, pipe2, 1))

# Start the threads
thread1.start()
thread2.start()

# Wait for both threads to complete
thread1.join()
thread2.join()

print("Image generation completed on both GPUs!")