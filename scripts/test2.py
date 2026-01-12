from PIL import Image
import numpy as np

img = Image.open("data/rice_split/test/Healthy Rice Leaf/Healthy_rice_leaf (40).jpg")
img = img.convert("RGB").resize((160, 160))
arr = np.array(img)

print(f"First pixel RGB: {arr[0,0]}")
print(f"First pixel processed: {(arr[0,0,0] / 127.5) - 1.0}")