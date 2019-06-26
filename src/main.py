import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from helpers import read_mnist, pca_fit, validate_model, timing
from sklearn.model_selection import learning_curve
from preprocess import deskew_vectorize

# Prepare data
train_X, train_y, val_X, val_y, test_X, test_y = read_mnist('../data/mnist.pkl.gz')

print('Use PCA to reduce data dimension...')
print('Initial data shape:', train_X.shape[1])
pca = pca_fit(X=train_X, infomation_presever=.95)
X_train = pca.transform(train_X)
X_val= pca.transform(val_X)
y_train = train_y
y_val = val_y
print('Data shape after reduce to 95%:', X_train.shape[1])

configs = [
    {'C':1.0, 'kernel':'rbf', 'gamma':'scale'},
    {'C':3.0, 'kernel':'rbf', 'gamma':'scale'},
    {'C':5.0, 'kernel':'rbf', 'gamma':'scale'},
    {'C':10.0, 'kernel':'rbf', 'gamma':'scale'},
    {'C':1.0, 'kernel':'rbf', 'gamma':.05},
    {'C':3.0, 'kernel':'rbf', 'gamma':.05},
    {'C':5.0, 'kernel':'rbf', 'gamma':.05},
    {'C':10.0, 'kernel':'rbf', 'gamma':.05},
    {'C':1.0, 'kernel':'poly', 'degree':3, 'gamma':'scale', 'coef0':0.0},
    {'C':3.0, 'kernel':'poly', 'degree':3, 'gamma':'scale', 'coef0':0.0},
    {'C':5.0, 'kernel':'poly', 'degree':3, 'gamma':'scale', 'coef0':0.0},
    {'C':10.0, 'kernel':'poly', 'degree':3, 'gamma':'scale', 'coef0':0.0},
]


print(f"{'~'*40}\nStart running validation with a subset of dataset...")
best_score = 0
best_config = None
for config in configs:
    score, model = validate_model(config, X_train, y_train, X_val, y_val)
    if score > best_score:
        best_score = score
        best_config = config
print(f"The best model with respect to config\n----{best_config}\nwhich have validation score: {best_score}")
print(f"{'~'*40}\nStart trainning 'best' model after validation with all dataset")

print(f"{'~'*40}\nCombine test set and validation set to train...")
total_train_X = np.r_[train_X, val_X]
total_train_y = np.r_[train_y, val_y]

print(f"{'~'*40}\nDeskew image for better result...")
deskewed_train = deskew_vectorize(total_train_X)
deskewed_test = deskew_vectorize(test_X)

print("The final result..................🤔🤔🤔❓")
score, clf = validate_model(best_config, deskewed_train, total_train_y, deskewed_test, test_y, 60_000, 10_000, print_score=False)
print("===========>", score)

# Plot some missclassify image
print("About to plot some wrong classify sample, control+C to stop...")
predict_y = clf.predict(deskewed_test)
idx = np.argwhere(predict_y != test_y)
plt.xticks([])
plt.yticks([])
for i in idx:
    plt.imshow(test_X[i].reshape(28, 28), cmap="gray")
    plt.title(f"Correct label {test_y[i]}, predicted label {predict_y[i]}")
    plt.show()
