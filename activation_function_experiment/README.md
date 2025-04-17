MNIST handwritten number classifier experiment

📚 This experiment is to see what effect can different activation function affect the model accuracy

## 🧪 DataSet
1. training data 60000 pictures
2. testing data  10000 pictures
3. size is all 28x28

## 🛠️ Model Structure
1. 3 hidden layer
2. epochs is 5/10/20 ( over 20 will cause overfitting )
3. optimizer is Adam (learning rate is 0.001)
4. training for 10 times to reduce unpredicted or unstable error

## 📈 Explanation for diagrams
1. relu         -> only use relu as activation function
2. sigmoid      -> only use sigmoid as activation function
3. relu_sigmoid -> use relu in first layer and rest is sigmoid as activation function

## 📌 Result
1. If only use relu as only activation function, the validation is not stable
2. Use relu and sigmoid can increase accuracy 
