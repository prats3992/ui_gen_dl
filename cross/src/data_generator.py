import os
import tensorflow as tf
import numpy as np


class DataGenerator:
    def __init__(self, data_path, image_size=(64, 64, 3), batch_size=32, categories=None):
        """
        Initialize data generator for directory-based image dataset

        Args:
            data_path (str): Root directory containing category subdirectories
            image_size (tuple): Desired image size (height, width, channels)
            batch_size (int): Batch size for training
            categories (list, optional): List of specific categories to use
        """
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size

        # Discover and process categories
        self.categories = self._get_categories(categories)

        # Generate image paths and labels
        self.image_paths, self.labels = self._load_image_paths()

        # Print dataset information
        self._print_dataset_info()

    def _get_categories(self, specified_categories=None):
        """
        Get list of categories (subdirectories)

        Args:
            specified_categories (list, optional): Specific categories to use

        Returns:
            list: List of category names
        """
        # Get all subdirectories
        categories = [
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]

        # Filter categories if specified
        if specified_categories:
            categories = [
                cat for cat in categories
                if cat in specified_categories
            ]

        # Sort categories for consistent label mapping
        return sorted(categories)

    def _load_image_paths(self):
        """
        Load image paths and their corresponding labels

        Returns:
            tuple: (list of image paths, list of labels)
        """
        image_paths = []
        labels = []

        for category_idx, category in enumerate(self.categories):
            category_path = os.path.join(self.data_path, category)

            # Get all image files in the category directory
            category_images = [
                os.path.join(category_path, img)
                for img in os.listdir(category_path)
                if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]

            # Add image paths and corresponding labels
            image_paths.extend(category_images)
            labels.extend([category_idx] * len(category_images))

        return image_paths, labels

    def _print_dataset_info(self):
        """
        Print information about the loaded dataset
        """
        print("Dataset Information:")
        for cat, cat_idx in zip(self.categories, range(len(self.categories))):
            cat_count = self.labels.count(cat_idx)
            print(f"Category {cat_idx}: {cat} - {cat_count} images")
        print(f"Total images: {len(self.image_paths)}")
        print(f"Number of categories: {len(self.categories)}")

    def decode_img(self, img_path):
        """
        Decode and preprocess an image

        Args:
            img_path (str): Path to the image file

        Returns:
            tf.Tensor: Preprocessed image tensor
        """
        # Read the image file
        img = tf.io.read_file(img_path)

        # Decode the image
        img = tf.io.decode_image(img, channels=3)

        img.set_shape([None, None, 3])

        # Resize the image
        img = tf.image.resize(
            img,
            [self.image_size[0], self.image_size[1]],
            method=tf.image.ResizeMethod.BILINEAR
        )

        # Convert to float32
        img = tf.cast(img, tf.float32)

        # Normalize to [-1, 1] range
        img = (img / 127.5) - 1.0

        return img

    def get_dataset(self, shuffle=True, augment=False):
        """
        Create TensorFlow dataset

        Args:
            shuffle (bool): Whether to shuffle the dataset
            augment (bool): Whether to apply data augmentation

        Returns:
            tf.data.Dataset: Prepared dataset
        """
        # Create dataset from image paths and labels
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.labels)
        )

        # Map images and labels
        def _process_path(img_path, label):
            img = self.decode_img(img_path)
            return img, label

        # Apply data augmentation if specified
        def _augment(img, label):
            if augment:
                # Random flip
                img = tf.image.random_flip_left_right(img)

                # Random brightness
                img = tf.image.random_brightness(img, max_delta=0.1)

                # Random contrast
                img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

            return img, label

        # Create dataset processing pipeline
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = (
            dataset
            .map(_process_path, num_parallel_calls=AUTOTUNE)
            .map(_augment, num_parallel_calls=AUTOTUNE)
        )

        # Shuffle and batch
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_paths))

        dataset = (
            dataset
            .batch(self.batch_size)
            .prefetch(AUTOTUNE)
        )

        return dataset

    @property
    def dataset(self):
        """
        Default dataset getter

        Returns:
            tf.data.Dataset: Dataset with default settings
        """
        return self.get_dataset()
