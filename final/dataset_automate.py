import json
import os
from collections import defaultdict

# Function to parse JSON and extract label types
def extract_labels(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    labels = defaultdict(int)
    for shape in data["shapes"]:
        labels[shape["label"]] += 1
    return labels, data

# Function to generate a descriptive input prompt
def generate_input(labels):
    prompt = "The screen includes "
    label_descriptions = []
    for label, count in labels.items():
        if count > 1:
            label_descriptions.append(f"{count} {label} elements")
        else:
            label_descriptions.append(f"1 {label} element")
    prompt += ", ".join(label_descriptions) + "."
    return prompt

# Function to compile data for fine-tuning
def compile_finetune_data(root_dir, output_file):
    dataset = []
    file_count = 0  # Counter for processed JSON files

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        # Ensure we are processing only directories
        if os.path.isdir(subdir_path):
            ui_type = subdir  # Use the subdirectory name as the UI type

            for json_file in os.listdir(subdir_path):
                if json_file.endswith(".json"):
                    json_path = os.path.join(subdir_path, json_file)
                    labels, data = extract_labels(json_path)

                    # Generate instruction and input
                    instruction = f"Generate JSON for a {ui_type.lower()} screen."
                    input_text = generate_input(labels)

                    # Add to dataset
                    dataset.append({
                        "instruction": instruction,
                        "input": input_text,
                        "output": data
                    })

                    # Increment the file count
                    # print(f"Processed {subdir}/{json_file}")
                    file_count += 1

    # Save compiled dataset
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Processed {file_count} JSON files.")

# Specify paths
root_dir = "./results"  # Root directory containing subdirectories for UI types
output_file = "compiled_finetune_data.json"

# Run the script
compile_finetune_data(root_dir, output_file)
print(f"Fine-tuning data compiled into {output_file}")
