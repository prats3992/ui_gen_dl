import json
from PIL import Image, ImageDraw, ImageFont

color_map = {
    "text": (211, 211, 211),
    "image": (0, 0, 0),
    "log_in": (0, 255, 255),
    "sign_up": (255, 0, 0),
    "username": (255, 0, 255),
    "password": (0, 255, 0),
    "icon": (0, 0, 255),
    "forgot": (255, 215, 0),
    "sm_button": (0, 191, 255),
    "button": (255, 20, 147),
    "box": (173, 255, 47),
    "privacy": (0, 0, 128),
    "check": (0, 250, 154),
    "name": (220, 20, 60),
    "navigation_dots": (240, 230, 140),
    "number": (30, 144, 255),
    "selector": (184, 134, 11),
    "search": (221, 160, 221),
    "edit_number": (34, 139, 34),
    "edit_string": (138, 43, 226),
    "filter": (139, 69, 19),
    "top_bar": (0, 139, 139),
    "heart_icon": (139, 0, 139),
    "sort": (250, 128, 114),
    "rating": (85, 107, 47),
    "bottom_bar": (176, 48, 96),
    "card_add": (143, 188, 143),
    "other": (72, 61, 139),
    "buy": (105, 105, 105),
    "default": (211, 211, 211)
}


def draw_element(draw, x1, y1, x2, y2, label, color_map):
    # Normalize label to lowercase for lookup
    label_lower = label.strip().lower()
    fill_color = color_map.get(label_lower, color_map["default"])
    draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline="purple", width=3)

def json_to_image(json_path, output_path, color_map):
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check which format we have
    if "shapes" in data:
        # New format:
        # imageWidth, imageHeight define the overall image size
        w = data.get("imageWidth", 800)
        h = data.get("imageHeight", 600)
        elements = data.get("shapes", [])

        # Each element has 'label' and 'points' = [[x1,y1],[x2,y2]]
        normalized_elements = []
        for elem in elements:
            label = elem.get("label", "")
            points = elem.get("points", [[0,0],[0,0]])
            # points are [[x1,y1],[x2,y2]]
            x1, y1 = points[0]
            x2, y2 = points[1]
            normalized_elements.append((x1, y1, x2, y2, label))

    else:
        # Original format:
        # Use top-level bounds for image size
        image_bounds = data.get("bounds", [0,0,800,600])
        w = image_bounds[2] - image_bounds[0]
        h = image_bounds[3] - image_bounds[1]

        # Child elements have 'componentLabel' and 'bounds'
        children = data.get("children", [])
        normalized_elements = []
        for child in children:
            component_label = child.get("componentLabel", "")
            bounds = child.get("bounds", [0,0,0,0])
            x1, y1, x2, y2 = bounds
            normalized_elements.append((x1, y1, x2, y2, component_label))

    # Create a blank white image
    image = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Draw elements
    for (x1, y1, x2, y2, label) in normalized_elements:
        draw_element(draw, x1, y1, x2, y2, label, color_map)

    # Save the resulting image
    image.save(output_path, "PNG")
    print(f"Image saved to: {output_path}")

if __name__ == "__main__":
    # Update these paths as needed
    json_file = "data.json"   # Replace with your JSON file path
    output_image = "output4.png"  # Replace with your desired output image name

    json_to_image(json_file, output_image, color_map)
