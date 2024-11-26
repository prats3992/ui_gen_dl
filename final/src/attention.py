import tensorflow as tf


class SelfAttnModel(tf.keras.Model):
    def __init__(self, input_dims, data_format='channels_last', **kwargs):
        super(SelfAttnModel, self).__init__(**kwargs)
        self.attn = _Attention(data_format=data_format)
        self.query_conv = tf.keras.layers.Conv2D(filters=input_dims // 16,
                                                 kernel_size=1,
                                                 data_format=data_format)
        self.key_conv = tf.keras.layers.Conv2D(filters=input_dims // 16,
                                               kernel_size=1,
                                               data_format=data_format)
        self.value_conv = tf.keras.layers.Conv2D(filters=input_dims,
                                                 kernel_size=1,
                                                 data_format=data_format)

    def build(self, input_shape):
        # No additional layers to build; just call super
        super(SelfAttnModel, self).build(input_shape)

    def call(self, inputs, training=False):
        q = self.query_conv(inputs)
        k = self.key_conv(inputs)
        v = self.value_conv(inputs)
        output, attention = self.attn([q, k, v, inputs])
        return output, attention  # Return only the output

    def compute_output_shape(self, input_shape):
        return input_shape  # Return the same shape as input


class _Attention(tf.keras.layers.Layer):
    def __init__(self, data_format='channels_last', **kwargs):
        super(_Attention, self).__init__(**kwargs)
        self.data_format = data_format

    def build(self, input_shapes):
        self.gamma = self.add_weight(
            name='gamma', shape=(), initializer=tf.initializers.Zeros)

    def call(self, inputs):
        if len(inputs) != 4:
            raise Exception('An attention layer should have 4 inputs.')

        query_tensor, key_tensor, value_tensor, origin_input = inputs
        input_shape = tf.shape(query_tensor)

        if self.data_format == 'channels_first':
            height_axis = 2
            width_axis = 3
        else:
            height_axis = 1
            width_axis = 2

        batchsize, height, width, _ = input_shape[0], input_shape[
            height_axis], input_shape[width_axis], input_shape[-1]

        proj_query = tf.reshape(
            query_tensor, (batchsize, height * width, -1))
        proj_key = tf.transpose(tf.reshape(
            key_tensor, (batchsize, height * width, -1)), perm=(0, 2, 1))
        proj_value = tf.reshape(
            value_tensor, (batchsize, height * width, -1))

        energy = tf.matmul(proj_query, proj_key)  # (batchsize, hw, hw)
        attention = tf.nn.softmax(energy, axis=-1)

        out = tf.matmul(attention, proj_value)  # (batchsize, hw, c)
        out = tf.reshape(out, (batchsize, height, width, -1))

        return tf.add(tf.multiply(out, self.gamma), origin_input), attention
