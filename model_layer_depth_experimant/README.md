MNIST handwritten number classifier experiment

📚 This experiment is to see what effect can different **layer** affect the model accuracy

## 🧪 DataSet
1. training data 60000 pictures
2. testing data 10000 pictures
3. size is all 28x28

## 🛠️ Model Structure
1. layer is set from 1, 3, 5, 7, 15
2. epochs is 5 ( to test for quick deploy situation )
3. optimizer is Adam (learning rate is 0.001)
4. activation function is relu for first layer, and the rest is sigmoid
5. training for 10 times to reduce unpredicted or unstable error

## 📌 Result
1. The deeper the model is, the accuracy might not be higher
