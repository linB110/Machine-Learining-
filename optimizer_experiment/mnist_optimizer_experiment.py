import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from matplotlib import pyplot as plt

# reproducibility
def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

# data loading
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train_flat = x_train.reshape(-1, 28*28)
x_test_flat = x_test.reshape(-1, 28*28)

# model builder
def build_model(hidden_layers, optimizer_name="adam"):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))
    for i, units in enumerate(hidden_layers):
        activation = 'relu' if i == 0 else 'sigmoid'
        model.add(keras.layers.Dense(units, activation=activation))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # 選擇 Optimizer
    optimizers = {
        "adam": Adam(learning_rate=0.001),
        "sgd": SGD(learning_rate=0.01),
        "rmsprop": RMSprop(learning_rate=0.001),
        "adagrad": Adagrad(learning_rate=0.01),
    }

    model.compile(
        optimizer=optimizers[optimizer_name],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# config 固定為 3-layer
depth_config = [128, 64, 32]

# 比較不同 optimizer
optimizers_to_test = ["adam", "sgd", "rmsprop", "adagrad"]

EPOCHS = 5
RUNS = 20

for opt_name in optimizers_to_test:
    print(f"\n🔧 Optimizer: {opt_name}")
    val_accuracies = []

    for run in range(RUNS):
        print(f"  🔁 Run {run+1}/{RUNS}", end='\r')
        set_seed(run)
        model = build_model(depth_config, optimizer_name=opt_name)
        history = model.fit(
            x_train_flat, y_train,
            validation_data=(x_test_flat, y_test),
            epochs=EPOCHS,
            batch_size=64,
            verbose=0
        )
        val_accuracies.append(history.history['val_accuracy'])

    val_accuracies = np.array(val_accuracies)
    mean_acc = np.mean(val_accuracies, axis=0)
    std_acc = np.std(val_accuracies, axis=0)

    # 畫圖
    epochs_range = np.arange(1, EPOCHS + 1)
    plt.plot(epochs_range, mean_acc, label=opt_name)
    plt.fill_between(epochs_range, mean_acc - std_acc, mean_acc + std_acc, alpha=0.15)

# final plot
plt.title(f'Validation Accuracy for Different Optimizers (avg of {RUNS} runs)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
