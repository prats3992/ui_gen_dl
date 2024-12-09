import tensorflow as tf


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, input_dims, output_dims, num_heads=1, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.output_head_dim = output_dims // num_heads

    def build(self, input_shape):
        # Create Dense layers for query, key, and value projections
        self.query_dense = tf.keras.layers.Dense(self.output_dims)
        self.key_dense = tf.keras.layers.Dense(self.output_dims)
        self.value_dense = tf.keras.layers.Dense(self.output_dims)

        # Add a projection layer to match the original input shape
        self.projection_layer = tf.keras.layers.Conv2D(
            filters=self.input_dims, kernel_size=1, padding='same'
        )

        super().build(input_shape)

    def call(self, inputs):
        query, value = inputs  # Query and value inputs

        # Get image dimensions
        batch_size = tf.shape(query)[0]
        height = tf.shape(query)[1]
        width = tf.shape(query)[2]
        channels = tf.shape(query)[3]

        # Flatten spatial dimensions for attention
        query_flat = tf.reshape(query, [batch_size, height * width, channels])
        value_flat = tf.reshape(value, [batch_size, height * width, channels])

        # Apply dense projections
        query = self.query_dense(query_flat)
        key = self.key_dense(value_flat)
        value = self.value_dense(value_flat)

        # Reshape for multi-head attention
        query = tf.reshape(
            query, [batch_size, height * width, self.num_heads, self.output_head_dim])
        key = tf.reshape(
            key, [batch_size, height * width, self.num_heads, self.output_head_dim])
        value = tf.reshape(
            value, [batch_size, height * width, self.num_heads, self.output_head_dim])

        # Transpose for attention computation
        # [batch, heads, seq, head_dim]
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        # Compute scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.output_head_dim, tf.float32))
        attention_scores = tf.matmul(query, key, transpose_b=True) / scale

        # Apply softmax to get attention probabilities
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)

        # Compute weighted sum of values
        context = tf.matmul(attention_probs, value)

        # Transpose and reshape back
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(
            context, [batch_size, height * width, self.output_dims])

        # Reshape back to original spatial dimensions
        context = tf.reshape(
            context, [batch_size, height, width, self.input_dims])

        # Project back using 1x1 convolution
        context = self.projection_layer(context)

        return context, attention_probs

    def compute_output_shape(self, input_shape):
        # Compute output shape
        query_shape, value_shape = input_shape
        return [(query_shape[0], query_shape[1], query_shape[2], self.input_dims),
                (query_shape[0], query_shape[1], query_shape[1], self.num_heads)]
