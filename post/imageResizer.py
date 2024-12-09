from PIL import Image

# Load the image
# Replace with the path to your image
image_path = "./samples/epoch_2000_class_0.png"
image = Image.open(image_path)

new_size = (360, 576)  # Width, Height
# Using ANTIALIAS for high-quality resizing
resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

# Save or display the resized image
resized_image.save("resized_image.jpg")  # Saves the resized image
print(f"The image has been resized to {
      new_size} pixels and saved as 'resized_image.jpg'.")
