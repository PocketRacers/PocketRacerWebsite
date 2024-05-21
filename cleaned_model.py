#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Lambda
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.losses import mean_squared_error
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def total_steering_loss(y_true, y_pred):
    n = tf.cast(tf.shape(y_true), dtype=tf.float32)
    f = lambda x: tfd.Independent(tfd.Cauchy(loc=x, scale=1.))
    kde_true = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[1 / n] * n),
        components_distribution=f(y_true)
    )
    kde_pred = tfd.MixtureSameFamily(
        reparameterize=False,
        mixture_distribution=tfd.Categorical(probs=[1 / n] * n),
        components_distribution=f(y_pred)
    )
    mean1 = kde_true.mean()
    mean2 = kde_pred.mean()
    steering_loss = tf.reduce_mean((mean1 - mean2) ** 2)
    return steering_loss

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )   
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection.units,
            "position_embedding_dim": self.position_embedding.output_dim,
        })
        return config
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_transformer_itdm(w, h, d, s, s_start, s_end, batch_size):
    patch_size = w
    num_patches = 28
    projection_dim = num_patches
    num_heads = 5
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 2
    mlp_head_units = [64, 64]

    image_inputs = Input(shape=(h, w, d), name='image_input')
    patches = Patches(patch_size)(image_inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)
    angle_out = Dense(1, activation='linear', name='angle_out')(features)

    model = Model(inputs=[image_inputs], outputs=[angle_out])
    optimizer = optimizers.Adam(lr=0.05)
    model.compile(loss={'angle_out': total_steering_loss},
                  optimizer=optimizer,
                  metrics={'angle_out': [total_steering_loss, 'mse']}, loss_weights=[1.0])
    model.summary()
    return model

def build_transformer3d(w, h, d, s):
    patch_size = 32
    num_patches = 32
    projection_dim = num_patches
    num_heads = 5
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 2
    mlp_head_units = [64, 64]

    image_inputs = Input(shape=(h, w, d), name='image_input')
    patches = Patches(patch_size)(image_inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)
    angle_out = Dense(1, activation='linear', name='angle_out')(features)

    model = Model(inputs=[image_inputs], outputs=[angle_out])
    optimizer = optimizers.Adam(lr=0.05)
    model.compile(loss={'angle_out': mean_squared_error},
                  optimizer=optimizer,
                  metrics={'angle_out': ['mse']}, loss_weights=[1.0])
    model.summary()
    return model

def build_transformer(w, h, d, s):
    patch_size = w
    num_patches = w
    projection_dim = num_patches
    num_heads = 5
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 3
    mlp_head_units = [128, 128]

    image_inputs = Input(shape=(h, w, d), name='image_input')
    patches = Patches(patch_size)(image_inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)
    angle_out = Dense(1, activation='linear', name='angle_out')(features)

    model = Model(inputs=[image_inputs], outputs=[angle_out])
    optimizer = optimizers.Adam(lr=0.01)
    model.compile(loss={'angle_out': mean_squared_error},
                  optimizer=optimizer,
                  metrics={'angle_out': ['mse']}, loss_weights=[1.0])
    model.summary()
    return model
