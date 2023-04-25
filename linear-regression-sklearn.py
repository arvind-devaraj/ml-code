#import the model
from sklearn import linear_model
reg = linear_model.LinearRegression()
# use it to fit a data
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
# Let's look into the fitted data
print(reg.coef_)
