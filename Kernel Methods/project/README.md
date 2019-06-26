# Kernel Methods for machine learning


The goal of the this data challenge is to implement machine learning algorithms from scratch and to acquire a better understanding of these techniques by adapting them to structural data. We consider a sequence classification task: predicting whether or not a DNA sequence region is a binding site to a specific transcription factor.
## Installation
This project runs on python >= 3.6, use pip to install dependencies:
```
pip3 install -r requirements.txt
```
## Usage
Use the `main.py` script to train a classifier using a kernel of your choice. The available classifiers are: SVM and logistics regression. The available kernels are: Gaussian kernel, Spectrum kernel and mismatch kernel. An example of use is as follows:
```
python3 main.py -c svm -k spectrum
>>> Please, Give a value for kernel parameter `spectrum`: 5
>>> Please, Give a value for classifier parameter `_lambda`: 0.01
```

  

It should be noted that after launching the program, it will ask you to choose values for the hyperparameters of your classifier and kernel. The predictions will be saved in the dumps folder.

  

To generate our best results, run the `start.py` script as follows:
```
python3 start.py
```