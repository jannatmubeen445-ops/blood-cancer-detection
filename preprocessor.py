import os
import cv2

# Main folders
original_folder = "Original"
output_folder = "Preprocessed"

# Output folder create karo
os.makedirs(output_folder, exist_ok=True)

# Original ke andar ke folders
for category in os.listdir(original_folder):

    input_category = os.path.join(original_folder, category)
    output_category = os.path.join(output_folder, category)

    # Sirf folders process karo
    if not os.path.isdir(input_category):
        continue

    # Output mein same category folder banao
    os.makedirs(output_category, exist_ok=True)

    # Category ke andar images
    for filename in os.listdir(input_category):

        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

            input_path = os.path.join(input_category, filename)

            image = cv2.imread(input_path)

            if image is None:
                print(f"Could not read: {input_path}")
                continue

            # Image ko 224 x 224 resize karo
            image = cv2.resize(image, (224, 224))

            # Slight blur for noise reduction
            image = cv2.GaussianBlur(image, (3, 3), 0)

            # Processed image save karo
            output_path = os.path.join(output_category, filename)
            cv2.imwrite(output_path, image)

            print(f"Processed: {category}/{filename}")

print("\nPreprocessing completed!")