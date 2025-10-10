import json
import sys
import doitlib
import numpy as np
from PIL import Image

def save_x_to_image(x_final, filename):
    # Normalize x_final to be in the valid image range [0, 255]
    x_final = (x_final - x_final.min()) / (x_final.max() - x_final.min())  # Normalize to [0,1]
    x_final = (x_final * 255).astype(np.uint8)  # Scale to [0,255]
    # Check the shape and adjust for grayscale images (MNIST)
    if x_final.shape[0] == 1:  # Remove batch dimension if present
        x_final = x_final[0]
      
    if x_final.shape[-1] == 1:  # MNIST has an extra channel dimension (28,28,1)
        x_final = x_final.squeeze(-1)  # Remove channel dimension to get (28,28)
    # Convert to PIL image
    image_mode = "L" if x_final.ndim == 2 else "RGB"  # 'L' for grayscale, 'RGB' for color
    image = Image.fromarray(x_final, mode=image_mode)
    # Create a unique temporary file in the current directory
    image.save(filename)
    print(f"Image saved {filename}")

if len(sys.argv) != 4:
    print(f"Usage {sys.argv[0]} dataset input_size cex_json_file\n")
    sys.exit(1)

dataset = sys.argv[1]
input_size = int(sys.argv[2])
cex_json = sys.argv[3]

x_test, y_test = doitlib.load_test_data(input_size=input_size, dataset=dataset)
with open(cex_json, 'r') as f:
    data = json.load(f)
for cex in data:
    idx = cex["index"]
    x = x_test[idx]
    save_x_to_image(x,f"orig_{dataset}_{idx}.png")
