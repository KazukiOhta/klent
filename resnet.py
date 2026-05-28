import keras
from keras import layers, initializers

# ResidualBlockV2 as a functional block
def ResidualBlockV2(inputs, num_channels, name="ResidualBlockV2"):
    x = layers.BatchNormalization(name=f"{name}_bn1")(inputs)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(num_channels, kernel_size=3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    x = layers.Conv2D(num_channels, kernel_size=3, padding="same", name=f"{name}_conv2")(x)
    x = layers.Add(name=f"{name}_add")([inputs, x])
    return x


# PQNet using Functional API
def PQNet(input_shape, num_actions, zero_init: bool, num_channels, num_blocks, name="PQNetV2"):
    inputs = keras.Input(shape=input_shape, name="input_layer")
    x = layers.Conv2D(num_channels, kernel_size=3, padding="same", name="conv_init")(inputs)
    
    # Add residual blocks
    for i in range(num_blocks):
        x = ResidualBlockV2(x, num_channels, name=f"block_{i}")

    x = layers.BatchNormalization(name="bn_after")(x)
    x = layers.Activation("relu", name="relu_after")(x)

    # Policy head
    policy_x = layers.Conv2D(2, kernel_size=1, padding="same", name="policy_conv")(x)
    policy_x = layers.BatchNormalization(name="policy_bn")(policy_x)
    policy_x = layers.Activation("relu", name="policy_relu")(policy_x)
    policy_x = layers.Flatten(name="policy_flatten")(policy_x)
    logits = layers.Dense(
        num_actions, name="logits", kernel_initializer=initializers.Zeros() if zero_init else None
    )(policy_x)

    # Q-Value head
    qvalue_x = layers.Conv2D(2, kernel_size=1, padding="same", name="qvalue_conv")(x)
    qvalue_x = layers.BatchNormalization(name="qvalue_bn")(qvalue_x)
    qvalue_x = layers.Activation("relu", name="qvalue_relu")(qvalue_x)           
    qvalue_x = layers.Flatten(name="qvalue_flatten")(qvalue_x)
    qvalue = layers.Dense(
        num_actions, name="qvalue", kernel_initializer=initializers.Zeros() if zero_init else None,
        activation="tanh",
    )(qvalue_x)
        
    # Build model
    model = keras.Model(
        inputs=inputs, 
        outputs={"logits": logits, "qvalue": qvalue}, 
        name=name
    )
    return model
