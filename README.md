# A Study on the Effectiveness of the Class Imbalance of Training Data on Convolutional Networks(CNNs)
Experiment about class imbalance problems inside the CNNs

# Prerequistes
1. pytorch, python : pytorch 1.0 ↑, python 3.7 ↑
2. package : numpy, os

# Measures Definition
1. Neuron Membership
2. Major class actvation - Minor class activation
3. Class Selectivity

# Dataset / Model
1. Data : Cifar10 with different imbalance ratio
2. Model : Vgg11 with batch normalization

# Experiment Result
1. Neuron Membership 
<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src = 'images/CL_cifar100_result.png' height = '400px'></td>
</tr>
</table>

2. Major class actvation - Minor class activation 
<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src = 'images/CL_cifar100_result.png' height = '400px'></td>
</tr>
</table>

3. Class Selectivity
<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src = 'images/CL_cifar100_result.png' height = '400px'></td>
</tr>
</table>
