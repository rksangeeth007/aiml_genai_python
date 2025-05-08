import tensorflow as tf

# maintain consistent performance
tf.random.set_seed(1)

# confirm GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.datasets import fashion_mnist

# load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
import numpy as np

# merge the training and testing sets
dataset = np.concatenate([x_train, x_test], axis=0)
# normalize the images from [0,255] to [0,1]
dataset = np.expand_dims(dataset, -1).astype("float32") / 255

BATCH_SIZE = 64

# convolution layers work 3 channels
dataset = np.reshape(dataset, (-1, 28, 28, 1))
# create a tensorflow dataset object
dataset = tf.data.Dataset.from_tensor_slices(dataset)
# set the batch size otherwise it reads one image at a time
dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

from tensorflow import keras
from tensorflow.keras import layers

# the generator's input is a noise vector
# hyper-parameter that also requires fine-tuning
NOISE_DIM = 150

generator = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(NOISE_DIM,)),
    layers.Dense(7*7*256),
    layers.Reshape(target_shape=(7, 7, 256)),
    layers.Conv2DTranspose(256, 3, activation="LeakyReLU", strides=2, padding="same"),
    layers.Conv2DTranspose(128, 3, activation="LeakyReLU", strides=2, padding="same"),
    layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")
])

generator.summary()

# design a discriminator with downsampling layers
discriminator = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(256, 3, activation="relu", strides=2, padding="same"),
    layers.Conv2D(128, 3, activation="relu", strides=2, padding="same"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

discriminator.summary()

optimizerG = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5)
optimizerD = keras.optimizers.Adam(learning_rate=0.00003, beta_1=0.5)

# binary classifier (real or fake)
lossFn = keras.losses.BinaryCrossentropy(from_logits=True)

# accuracy metric
gAccMetric = tf.keras.metrics.BinaryAccuracy()
dAccMetric = tf.keras.metrics.BinaryAccuracy()

@tf.function
def trainDStep(data):
    # the batch is (32,28,28,1), so extract 32 value
    batchSize = tf.shape(data)[0]
    # create a noise vector as generator input sampled from Gaussian Random Normal
    # As an exercise try sampling from a uniform distribution and observe the difference
    noise = tf.random.normal(shape=(batchSize, NOISE_DIM))

    # concatenate the real and fake labels
    y_true = tf.concat(
        [
            # the original data is real, labeled with 1
            tf.ones(batchSize, 1),
            # the forged data is fake, labeled with 0
            tf.zeros(batchSize, 1)
        ],
        axis=0
    )

    # record the calculated gradients
    with tf.GradientTape() as tape:
        # generate forged samples
        fake = generator(noise)
        # concatenate real data and forged data
        x = tf.concat([data, fake], axis=0)
        # see if the discriminator detects them
        y_pred = discriminator(x)
        # calculate the loss
        discriminatorLoss = lossFn(y_true, y_pred)

    # apply the backward path and update weights
    grads = tape.gradient(discriminatorLoss, discriminator.trainable_weights)
    optimizerD.apply_gradients(zip(grads, discriminator.trainable_weights))

    # report accuracy
    dAccMetric.update_state(y_true, y_pred)

    # return the loss for visualization
    return {
        "discriminator_loss": discriminatorLoss,
        "discriminator_accuracy": dAccMetric.result()
    }


@tf.function
def trainGStep(data):
    batchSize = tf.shape(data)[0]
    noise = tf.random.normal(shape=(batchSize, NOISE_DIM))
    # when training the generator, we want it to maximize the probability that its
    # output is classified as real, remember the min-max game
    y_true = tf.ones(batchSize, 1)

    with tf.GradientTape() as tape:
        y_pred = discriminator(generator(noise))
        generatorLoss = lossFn(y_true, y_pred)

    grads = tape.gradient(generatorLoss, generator.trainable_weights)
    optimizerG.apply_gradients(zip(grads, generator.trainable_weights))

    gAccMetric.update_state(y_true, y_pred)

    return {
        "generator_loss": generatorLoss,
        "generator_accuracy": gAccMetric.result()
    }

from matplotlib import pyplot as plt


def plotImages(model):
    images = model(np.random.normal(size=(81, NOISE_DIM)))

    plt.figure(figsize=(9, 9))

    for i, image in enumerate(images):
        plt.subplot(9,9,i+1)
        plt.imshow(np.squeeze(image, -1), cmap="Greys_r")
        plt.axis('off')

    plt.show();

for epoch in range(30):

    # accumulate the loss to calculate the average at the end of the epoch
    dLossSum = 0
    gLossSum = 0
    dAccSum = 0
    gAccSum = 0
    cnt = 0

    # loop the dataset one batch at a time
    for batch in dataset:

        # train the discriminator
        # remember you could repeat these 2 lines of code for K times
        dLoss = trainDStep(batch)
        dLossSum += dLoss['discriminator_loss']
        dAccSum += dLoss['discriminator_accuracy']

        # train the generator
        gLoss = trainGStep(batch)
        gLossSum += gLoss['generator_loss']
        gAccSum += gLoss['generator_accuracy']

        # increment the counter
        cnt += 1

    # log the performance
    print("E:{}, Loss G:{:0.4f}, Loss D:{:0.4f}, Acc G:%{:0.2f}, Acc D:%{:0.2f}".format(
        epoch,
        gLossSum/cnt,
        dLossSum/cnt,
        100 * gAccSum/cnt,
        100 * dAccSum/cnt
    ))

    if epoch % 2 == 0:
        plotImages(generator)

# generate some images with the trained model
# observe how fast it is compared to rendering an image with computer graphics algorithms
# that's why GANs can revolutionize the video games industry by generating realistic scenes
# observe how the generated samples seem to belong to the same or similar class; this is the
# "mode collapse problem" of GAN's.
images = generator(np.random.normal(size=(81, NOISE_DIM)))

# plot the generated samples
from matplotlib import pyplot as plt

plt.figure(figsize=(9, 9))

for i, image in enumerate(images):
    plt.subplot(9,9,i+1)
    plt.imshow(np.squeeze(image, -1), cmap="Greys_r")
    plt.axis('off')

plt.show();
