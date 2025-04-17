import numpy as np
from keras import Sequential
from keras.layers import Dense

# create training data and labels
data  = np.random.random(10000)
data.shape = 10000, 1
label = np.array(data >= 0.5, dtype = int)  # make label from true/false to 1/0

# test for data, label correctness
# for i in range(10):
#     print(data[i])
#     print(label[i])

# build model
model = Sequential ([
    Dense(units = 8, input_shape = (1,), activation = 'relu'),
    Dense(units = 2, activation = 'softmax')
]) 

# see model params number and other information
# print(model.summary())

# compile the model
model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

# train the model
model.fit(
    x = data,
    y = label,
    epochs = 10,
    verbose = 2
)

# test the model
test_data  = np.random.random(1000)
test_data.shape = 1000, 1
test_label = np.array(data >= 0.5, dtype = int)
prediction = model.predict(test_data)
# print(np.argmax(prediction, axis = 1))

# validation while training model
model.fit(
    x = data,
    y = label,
    epochs = 10,
    verbose = 2,
    validation_split = .1
)