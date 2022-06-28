import tensorflow as tf
import tensorflow_datasets as tfds
from IPython.display import clear_output
import matplotlib.pyplot as plt
from models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SCCE
from tensorflow.keras.callbacks import EarlyStopping
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation', type=str, help='Activation function for the U-Net')
    parser.add_argument('--modules', type=int, help='Number of modules per node')
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    print("\nActivation function: %s" % args.activation)

    model = unet_2plus((128, 128, 3), num_classes=3, kernel_size=3, modules_per_node=args.modules, pool_size=(2, 2), upsample_size=(2, 2), levels=3,
                filter_num=(16, 32, 64), activation=args.activation, batch_normalization=True, use_bias=False)
    model.summary()
    model.compile(loss=SCCE(from_logits=True), optimizer=Adam(learning_rate=1e-4), metrics='accuracy')


    def normalize(input_image, input_mask):
      input_image = tf.cast(input_image, tf.float32) / 255.0
      input_mask -= 1
      return input_image, input_mask


    def load_image(datapoint):
      input_image = tf.image.resize(datapoint['image'], (128, 128))
      input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

      input_image, input_mask = normalize(input_image, input_mask)

      return input_image, input_mask


    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)


    class Augment(tf.keras.layers.Layer):
      def __init__(self, seed=42):
        super().__init__()

      def call(self, inputs, labels):
        inputs = tf.image.random_flip_left_right(inputs, seed=42)
        labels = tf.image.random_flip_left_right(labels, seed=42)
        return inputs, labels


    train_batches = (
        train_images
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .map(Augment())
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(BATCH_SIZE)

    EPOCHS = 100
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    early_stopping = EarlyStopping('val_accuracy', patience=10, verbose=1, restore_best_weights=True)

    model_history = model.fit(train_batches, epochs=EPOCHS,
                              callbacks=[early_stopping],
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches,
                              verbose=True)

    def display(display_list):
      plt.figure(figsize=(15, 15))

      title = ['Input Image', 'True Mask', 'Predicted Mask']

      for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
      plt.show()

    def create_mask(pred_mask):
      pred_mask = tf.argmax(pred_mask, axis=-1)
      pred_mask = pred_mask[..., tf.newaxis]
      return pred_mask[0]

    def show_predictions(dataset=None, num=1):
      if dataset:
        for image, mask in dataset.take(num):
          pred_mask = model.predict(image)
          display([image[0], mask[0], create_mask(pred_mask)])

    # show_predictions(test_batches, 3)

    model.evaluate(test_batches)

    del model
