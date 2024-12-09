import numpy as np
import tensorflow as tf

from .cross_attention import CrossAttention


def create_generator(input_shape=(64, 64, 3), latent_dim=100, num_classes=5, num_heads=1):
    with tf.device('/GPU:0'):
        # Ensure input_shape is a tuple
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape, 3)

        if len(input_shape) == 2:
            input_shape = (*input_shape, 3)

        # Image input for conditioning
        image_input = tf.keras.Input(shape=input_shape)

        # Latent vector input
        latent_input = tf.keras.Input(shape=(latent_dim,))

        # Class label input
        label_input = tf.keras.Input(shape=(1,), dtype=tf.int32)

        # Embed label
        label_embedding = tf.keras.layers.Embedding(
            input_dim=num_classes,
            output_dim=latent_dim
        )(label_input)
        label_embedding = tf.keras.layers.Flatten()(label_embedding)

        # Combine latent vector with label embedding
        combined_input = tf.keras.layers.Concatenate()(
            [latent_input, label_embedding])

        # Process latent input through dense layers
        x = tf.keras.layers.Dense(64 * 64, activation='relu')(combined_input)
        x = tf.keras.layers.Reshape((64, 64, 1))(x)

        # Downsampling and feature extraction
        x = tf.keras.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same", activation="relu")(x)

        x = tf.keras.layers.BatchNormalization()(x)

        # CrossAttention with multi-heads
        cross_attn, attn_probs = CrossAttention(
            input_dims=x.shape[-1],
            output_dims=x.shape[-1],
            num_heads=num_heads
        )([x, x])

        # No need for reshape, cross_attn is now projected to match input dimensions
        x = tf.keras.layers.Add()([x, cross_attn])

        # Upsampling layers
        x = tf.keras.layers.Conv2DTranspose(
            64, kernel_size=4, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # Final image generation layer
        outputs = tf.keras.layers.Conv2D(
            3, kernel_size=3, padding="same", activation="tanh"
        )(x)

    return tf.keras.Model(
        [image_input, latent_input, label_input],
        [outputs, attn_probs, x],
        name="generator"
    )


def create_discriminator(input_shape=(64, 64, 3), num_heads=1, num_classes=5):
    with tf.device('/GPU:0'):
        # Ensure input_shape is a tuple
        if isinstance(input_shape, int):
            input_shape = (input_shape, input_shape, 3)

        if len(input_shape) == 2:
            input_shape = (*input_shape, 3)

        # Image input
        image_input = tf.keras.Input(shape=input_shape)

        # Label input
        label_input = tf.keras.Input(shape=(1,), dtype=tf.int32)

        # Embed label
        label_embedding = tf.keras.layers.Embedding(
            input_dim=num_classes,
            output_dim=input_shape[0] * input_shape[1]
        )(label_input)

        # Reshape label embedding to match image dimensions
        label_embedding = tf.keras.layers.Reshape(
            target_shape=(input_shape[0], input_shape[1], 1)
        )(label_embedding)

        # Repeat the single-channel label embedding to match image channels
        label_embedding = tf.keras.layers.Concatenate(axis=-1)([
            label_embedding] * input_shape[2]
        )

        # Combine image with label embedding
        x = tf.keras.layers.Concatenate(
            axis=-1)([image_input, label_embedding])

        # Initial convolution
        x = tf.keras.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        # CrossAttention
        cross_attn, attn1 = CrossAttention(
            input_dims=x.shape[-1],
            output_dims=x.shape[-1],
            num_heads=num_heads
        )([x, x])

        # No need for reshape, cross_attn is now projected to match input dimensions
        x = tf.keras.layers.Add()([x, cross_attn])

        # Flatten and classification
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)

        # Output layer
        output = tf.keras.layers.Dense(1, activation="linear")(x)

    return tf.keras.Model([image_input, label_input], [output, attn1], name="discriminator")
