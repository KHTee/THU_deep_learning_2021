# Logistic_Regression_with_MNIST_handwriting
 Implementation from scratch for classification using logistic regression for two classes of MNIST handwriting dataset with logistic regression. Script will auto download subset of MNIST handwriting dataset (3 & 6).

## Requirements

- python==3.8
- matplotlib
- scikit-learn
- numpy

```
pip install -r requirements.txt
```

## Training
```
python3 logit_reg.py --epoch 5000 --batch-size 128 --lr 0.01
```

## Output
Testing accuracy will be printed out in terminal. Accuracy and loss graphs will be plotted and saved in current directory.