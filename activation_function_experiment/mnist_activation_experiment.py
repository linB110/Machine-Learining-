import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

# make result reproducible
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

# load and pre-processing data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# build model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape=(784,), activation='sigmoid'),
        keras.layers.Dense(64, activation='sigmoid'),
        keras.layers.Dense(32, activation='sigmoid'),
        keras.layers.Dense(10, activation='softmax'),
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# experiment parameters 
EPOCHS = 20        # (change is epoch)
RUNS = 10          # run multiple times to reduce training unpredicted error
val_accuracies = []

# multiple training process
for run in range(RUNS):
    print(f"🔁 No.{run+1} training...")
    set_seed(run)  
    model = build_model()
    history = model.fit(
        x_train_flat, y_train,
        validation_data=(x_test_flat, y_test),
        epochs=EPOCHS,
        batch_size=64,
        verbose=0  # not show training process
    )
    val_accuracies.append(history.history['val_accuracy'])

val_accuracies = np.array(val_accuracies)
mean_acc = np.mean(val_accuracies, axis=0)
std_acc = np.std(val_accuracies, axis=0)

# plotting result
epochs_range = np.arange(1, EPOCHS+1)
plt.plot(epochs_range, mean_acc, label='Mean Validation Accuracy', linewidth=2)
plt.fill_between(epochs_range,
                 mean_acc - std_acc,
                 mean_acc + std_acc,
                 alpha=0.3, label='±1 Std Dev')
plt.title(f'Mean Validation Accuracy over {RUNS} Runs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
