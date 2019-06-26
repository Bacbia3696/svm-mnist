# SVM for MNIST dataset
> Simple python implementation with sklearn library for MNIST dataset, which achive more than 98% accuracy

## Fast validation
We use PCA to reduce input dimension from 784 to 154 but presever 95% information

We use only a subset of input set to validate to make validation process even faster (10_000 samples)

We expect to faster than $\frac{784}{154}*5^2 \approx 127 (times)$

## Preprocess
We use deskew method as preprocess method when we train the finally model for better accuracy


## Install dependencies with pip
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Note
This repository is quite heavy because we include mnist dataset file

The unittest haven't been written yet 😅😅

## References
[mnist_homepage](http://yann.lecun.com/exdb/mnist/)

[deskew](https://fsix.github.io/mnist/Deskewing.html)

[PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

[sklearn](https://scikit-learn.org/stable/)

## Images
[![asd]](./screenshots/trainning.png)

## Author
bacbia3696@gmail.com

1512587@student.hcmus.edu.vn

## License
[MIT](https://choosealicense.com/licenses/mit/)
