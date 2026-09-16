import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "tools"))
print("hello analyzing....")

from image_analysis import analyze_image

image_path = os.path.join(os.path.dirname(__file__), "..", "generated_images", "2594910462563289955.jpeg")

result = analyze_image(image_path, "What is shown in this image?")
print("Analyzing...")
print(result)