MNIST handwritten number classifier experiment

📚 This experiment is to how different **optimizer** affects the model accuracy

## 🧪 DataSet
1. training data 60000 pictures
2. testing data 10000 pictures
3. size is all 28x28

## 🛠️ Model Structure
1. layer is 3
2. epochs is 5/10 ( to test for quick deploy situation )
3. optimizer is Adam/SGD/RMSProp/Adagrad
4. activation function is relu for first layer, and the rest is sigmoid
5. training for 20 times to reduce unpredicted or unstable error

## 📌 Result
1. For both epochs = 5 or 10, Adam and RMSProp performs the best


