import cv2
import numpy as np
import os


def create_semantic_image(normal_image_path, annotations, output_path="semantic_image.jpg"):
    """
    Create a semantic representation of an image with bounding boxes and labels.

    Args:
        normal_image_path (str): Path to the input image
        annotations (list): List of dictionaries containing label and bounding box info
        output_path (str, optional): Path to save the semantic image

    Returns:
        numpy.ndarray: Semantic image with bounding boxes and labels
    """
    try:
        # Load the normal image
        img = cv2.imread(normal_image_path)

        if img is None:
            raise FileNotFoundError(
                f"Could not read image at {normal_image_path}")

        img_h, img_w, _ = img.shape

        # Create a blank image for the semantic representation (black background)
        semantic_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Color map for different element types
        color_map = {
            'button': (255, 0, 0),    # Blue
            'input': (0, 255, 0),      # Green
            'text': (0, 0, 255),       # Red
            'navbar': (255, 255, 0),   # Yellow
            'logo': (255, 0, 255),     # Magenta
            'default': (128, 128, 128)  # Gray
        }

        # Process each annotation
        for annotation in annotations:
            label = annotation.get('label', 'default')
            x1, y1, x2, y2 = annotation['bbox']

            # Choose color based on label (or default)
            color = color_map.get(label.lower(), color_map['default'])

            # Draw bounding box for the UI element
            cv2.rectangle(semantic_image, (x1, y1), (x2, y2), color, 2)

            # Write label text inside or above the bounding box
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x1
            text_y = max(y1 - 5, 0)  # Ensure text is within image bounds

            cv2.putText(semantic_image, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # Save the semantic image
        cv2.imwrite(output_path, semantic_image)

        return semantic_image

    except Exception as e:
        print(f"Error creating semantic image: {e}")
        return None


def main():
    # Simulate a UI screenshot annotation scenario

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    # Example annotations for a hypothetical login page
    login_page_annotations = [
        {
            'label': 'logo',
            'bbox': [50, 30, 250, 150]  # Top logo area
        },
        {
            'label': 'text',
            'bbox': [100, 200, 500, 250]  # Welcome text
        },
        {
            'label': 'input',
            'bbox': [100, 300, 500, 350]  # Username input
        },
        {
            'label': 'input',
            'bbox': [100, 400, 500, 450]  # Password input
        },
        {
            'label': 'button',
            'bbox': [250, 500, 400, 550]  # Login button
        },
        {
            'label': 'text',
            'bbox': [150, 600, 450, 650]  # Forgot password link
        }
    ]

    # Path to a sample image (you would replace this with your actual image)
    sample_image_path = './data/train/Akin_SAGAN_500/images/login/100.jpg'

    # Create semantic image
    semantic_image = create_semantic_image(
        sample_image_path,
        login_page_annotations,
        'output/login_page_semantic.jpg'
    )

    # Optionally, print out the annotations for reference
    print("Semantic Annotations:")
    for annotation in login_page_annotations:
        print(f"Label: {annotation['label']}, Bounding Box: {
              annotation['bbox']}")


if __name__ == "__main__":
    main()
