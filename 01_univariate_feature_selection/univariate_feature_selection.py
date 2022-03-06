from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X, y = load_iris(return_X_y=True)

print(X.shape)
print(f'Number of columns before selecting 2 best features is {X.shape[1]}')

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)
print(f'Number of columns REMAINING after selecting 2 best features is {X_new.shape[1]}')
