# 🧠 Machine Learning Experiments

This repository contains my learning and experiment records while studying **Machine Learning**, especially focused on training neural networks using Keras and TensorFlow.

## 📁 Project Structure

Each subfolder contains a different type of experiment with code and result visualizations:

- [`activation_function_experiment/`](./activation_function_experiment): Explore how different activation functions (ReLU, Sigmoid, ELU, etc.) affect performance.
- [`model_layer_depth_experimant/`](./model_layer_depth_experimant): Test the effect of neural network depth (1, 3, 5, 7, 15 layers) on accuracy and overfitting.
- [`optimizer_experiment/`](./optimizer_experiment): Compare optimizers like Adam, SGD, RMSProp, etc.

## 🧪 Baseline Model

Most experiments use a 3-layer fully connected model as baseline:
- Hidden Layers: `[128, 64, 32]`
- Activation: First layer ReLU, others Sigmoid
- Optimizer: Adam
- Dataset: MNIST

## 📊 Visualizations

Each experiment folder contains:
- Python code for training
- Accuracy graphs
- README explaining insights and results

## 🚀 To Run

```bash
pip install tensorflow numpy matplotlib
