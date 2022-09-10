# **Machine Learning Notes**

> # Data Preprocessing

## Importing the Libraries
- numpy
: To work with arrays

- matplotlib
: To plot graphs

- pandas
: To work with data matrices and vectors

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Importing the Dataset
| Country | Age  | Salary  | Purchased |
|---------|------|---------|-----------|
| France  | 44.0 | 72000.0 | No        |
| Spain   | 27.0 | 48000.0 | Yes       |
| Germany | 30.0 | 54000.0 | No        |
| Spain   | 38.0 | 61000.0 | No        |
| Germany | 40.0 | NaN     | Yes       |
| France  | 35.0 | 58000.0 | Yes       |
| Spain   | NaN  | 52000.0 | No        |
| France  | 48.0 | 79000.0 | Yes       |
| Germany | 50.0 | 83000.0 | No        |
| France  | 37.0 | 67000.0 | Yes       |
```python
dataset = pd.read_csv('../Path/Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(X)
print(y)
```

## Taking care of missing data
1. Exclude rows of missing Data
2. Replacing missing values to
    - Mean value
    - Median value
    - Mode value
### Importing other modules
- scikit-learn
: module for machine learning and data mining
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print(X)
```
[comment]: <> (Webapp for MD table https://www.tablesgenerator.com/markdown_tables)

| Country | Age  | Salary  | Purchased |
|---------|------|---------|-----------|
| France  | 44.0 | 72000.0 | No        |
| Spain   | 27.0 | 48000.0 | Yes       |
| Germany | 30.0 | 54000.0 | No        |
| Spain   | 38.0 | 61000.0 | No        |
| Germany | 40.0 | 63777.7 | Yes       |
| France  | 35.0 | 58000.0 | Yes       |
| Spain   | 38.8 | 52000.0 | No        |
| France  | 48.0 | 79000.0 | Yes       |
| Germany | 50.0 | 83000.0 | No        |
| France  | 37.0 | 67000.0 | Yes       |

## Encoding categorial data [Independent Variable]
Categorial data, needs to be encoded to numerical data type
- OneHotEncoder
: converts a categorial data column into sevral columns of 0 and 1
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
```
| Country0 | Country1 | Country1 | Age  | Salary   | Purchased |
|----------|----------|----------|------|----------|-----------|
| 1.0      | 0.0      | 0.0      | 44.0 | 72000.0  | No        |
| 0.0      | 0.0      | 1.0      | 27.0 | 48000.0  | Yes       |
| 0.0      | 1.0      | 0.0      | 30.0 | 54000.0  | No        |
| 0.0      | 0.0      | 1.0      | 38.0 | 61000.0  | No        |
| 0.0      | 1.0      | 0.0      | 40.0 | 63777.8  | Yes       |
| 1.0      | 0.0      | 0.0      | 35.0 | 58000.0  | Yes       |
| 0.0      | 0.0      | 1.0      | 38.8 | 52000.0  | No        |
| 1.0      | 0.0      | 0.0      | 48.0 | 79000.0  | Yes       |
| 0.0      | 1.0      | 0.0      | 50.0 | 83000.0  | No        |
| 1.0      | 0.0      | 0.0      | 37.0 | 67000.0  | Yes       |

## Encoding categorial data [Dependent Variable]
- Label Encoder
: Convert categorial data into binary
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
```
| Country0 | Country1 | Country2 | Age  | Salary   | Purchased |
|----------|----------|----------|------|----------|-----------|
| 1.0      | 0.0      | 0.0      | 44.0 | 72000.0  | 0         |
| 0.0      | 0.0      | 1.0      | 27.0 | 48000.0  | 1         |
| 0.0      | 1.0      | 0.0      | 30.0 | 54000.0  | 0         |
| 0.0      | 0.0      | 1.0      | 38.0 | 61000.0  | 0         |
| 0.0      | 1.0      | 0.0      | 40.0 | 63777.8  | 1         |
| 1.0      | 0.0      | 0.0      | 35.0 | 58000.0] | 1         |
| 0.0      | 0.0      | 1.0      | 38.8 | 52000.0  | 0         |
| 1.0      | 0.0      | 0.0      | 48.0 | 79000.0  | 1         |
| 0.0      | 1.0      | 0.0      | 50.0 | 83000.0  | 0         |
| 1.0      | 0.0      | 0.0      | 37.0 | 67000.0  | 1         |

## Spliting the dataset into Training and Test data
Split dataset into train and test data, to test the model

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

## Feature Scaling
Scaling Dataset, such that all the values lie inside a small range
- Standardisation
: This will scale values to [-3,+3]

[comment]: <> (Webapp for MD equation https://latex.codecogs.com/eqneditor/editor.php)
$$ X_{stand} = \frac{x- X_{mean}}{standard \ \ deviation(X)} $$

- Normalisation
: This will scale values to [0,1]

$$ X_{norm} = \frac{x- X_{mean}}{max(X)-min(X)} $$

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# fit using training data
sc.fit(X_train[:,3:])
# transform training data using the scaler
X_train[:,3:] = sc.transform(X_train[:,3:])
# transform test data using the same scaler
X_test[:,3:] = sc.transform(X_test[:,3:])
```

> # Simple Linear Regression
### Predicts continious numerical values
In simple linear regression, we simply predict value based on an equation
$$ y = b_{0} \ \ + \ \ b_{1}x $$
In above equation, y will depend on the values of x, so we can predict/calculate value of y, if value of x is already known.

> ### Loading new Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv("../Datasets/Regression/Simple_Linear_Regression/Salary_Data.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
```
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```
## Training the Simple Regression Model on the Training dataset
In this step, Simple Linear Regression model is trained using the training dataset.   
Simply equation for the Linear Regression is computed, using training dataset.
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
## Predicting the result for Test Dataset
In this step, result for test data is predicted using above trained model.
```python
y_pred = regressor.predict(X_test)
print(y_pred)
print(y_test)
```
## Visualising the Training set result
```python
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylable('Salary')
plt.show()
```
## Visualising the Test set result
```python
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylable('Salary')
plt.show()
```
## Predict salary for a single Experience
```python
print(regressor.predict([[0]]))
```
## Computing the Linear Regression Equation
```python
n=len(X_train[0])
ans=[]
b0=regressor.predict([[0  for _ in range(n)]])
eq="y = "+str(b0[0])
for i in range(n):
    ar=[[0 for _ in range(i)]+[1]+[0 for _ in range(i+1,n)]]
    b=regressor.predict(ar)
    eq+=" + {}*x{}".format(b[0],i+1)
print(eq)
```
> # Multiple Linear Regression
If there are multiple variables inside Linear Regression Equation, it is known as Multiple Linear Regression
Let's see equation

$$y = b_{0} \ \ + \ \ b_{1}x_{1}  \ \ + \ \ b_{2}x_{2}  \ \ + \ \ b_{3}x_{3} \ \ + \ \ ... \ \ + \ \ b_{n}x_{n}$$

Now y will depend on multiple values, we can still predict/calculate value of y, if we have the equation and the values of the X.

## Assumptions of Linear Regression
There are 5 assumptions
1. Linearity
2. Homoscedasticity
3. Multivariate Normality
4. Independence of Errors
5. Lack of Multicollinearity

## Hypothesis Testing (P-Value)
Supose you have a coin, and you are tossing it continiously
1. got HEAD, its probability is 0.5
2. got again HEAD, its probability is 0.25
3. got again HEAD, its probability is 0.12
4. got again HEAD, its probability is 0.06

If you get get HEAD again (probability 0.03), coin is suspicous. you may think coin is not a fair coin.\
The point/percent at which you become suspicious is known as **Significance Value**.\
Initially we assume coin is fair, but when the significance value is reached, we correct our assumption (Hypothesis) and confirm that it is not a fair coin, this is known as **Null Hypothesis Testing**.

## Building a Multiple Linear Regression Model
There are 5 ways to build Multiple Linear Regression Model
1. All In
:  Select all the columns
    - You have prior knowledge
    - You have to select all columns
    - You are preparing for backward elimination
2. Backward Elimination (Fastest)
    1. Select a Significance Level to stay in the model (SL=0.05)
    2. Fit the model with all columns/predictor
    3. Select the column/predictor with highest P-Value, if P-Value > SL goto step 4, else go to FINISH
    4. Remove the predictor
    5. Fit the model again, and go to step 3
3. Forward Selection
    1. Select a Significance Level to Enter in the model (SL=0.05)
    2. Fit the model with all predictors seperately
    3. Select the predictor with lowest P-value, if P-value < SL goto step 4, else FINISH and keep the last model
    4. Fit the model, using all the remaining predictors separately and with this one extra combination
4. Bidirection Elemination
    1. Select a Significance Level to stay in the model (STAY_SL=0.05) and a Significance Level to Enter in the model (ENTER_SL=0.05)
    2. Perform next step of Forward Selection
    3. Perform all step of Backward Elimination
    4. When no new predictors enter and no old predictors can exit, FINISH
5. Score Comparision
    1. Select a criterion of goodness (Score)
    2. Construct all possible regression model (2<sup>N</sup>-1)
    3. Select the best model using goodness criterion
    4. FINISH

Step-wise Regression
- Backward Elimination
- Forward Selection
- Bidirection Elemination

## Loading Data and preprocessing it
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Regression/Multiple_Linear_Regression/50_Startups.csv')
# print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Taking care of Missing values
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,:-1])
X[:,:-1] =  imputer.transform(X[:,:-1])
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(X[:,-2:-1])
X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# print(X)

# Encoding categorial Data [One Hot Encoding]
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```
## Training the Linear Regression Model on the training set
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

## Predicting the result for test set result
```python
y_pred = regressor.predict(X_test)
```

## Visualising the predicted and actual results
```python
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),axis=1))
```

## Computing the equation of model
```python
n = len(X_train[0])
b0=regressor.predict([[0 for _ in range(n)]])
eq="y = {}".format(b0[0])
for i in range(n):
    eq+=" + {}*x{}".format(regressor.predict([[0 for _ in range(i)]+[1]+[0 for _ in range(i+1,n)]])[0],i+1)
print(eq)
```

> # Polynomial Regression
If the power raise to variable is not only 1, but may have different powers of the variable x, it is known as Polynomial Regression.
Let's see the equation
$$y = b_{0} \ \ + \ \ b_{1}x^{1}  \ \ + \ \ b_{2}x^{2}  \ \ + \ \ b_{3}x^{3} \ \ + \ \ ... \ \ + \ \ b_{n}x^{n}$$
Now y will depend on multiple powers of the x, we can still predict/calculate value of y, if we have the equation and the value of the x.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Regression/Polynomial_Regression/Position_Salaries.csv')
# print(dataset)

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

# Taking care of Missing values
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:,:-1])
# X[:,:-1] =  imputer.transform(X[:,:-1])
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X[:,-2:-1])
# X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# print(X)

# Encoding categorial Data [One Hot Encoding]
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# # print(X)

# # Splitting dataset into train and test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

## Training the linear regression model on whole dataset
```python
from sklearn.linear_model import LinearRegression
lin_reg_1 =  LinearRegression()
lin_reg_1.fit(X,y)
```

## Training the polynomial regression model on whole dataset [n=2]
```python
from sklearn.preprocessing  import PolynomialFeatures
poly_fet = PolynomialFeatures(degree=2)
poly_X = poly_fet.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_X,y)
```

## Visualising the Linear Regression result
```python
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_1.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression Model)')
plt.show()
```

## Visualising the Polynomial Regression result
```python
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_X),color='blue')
plt.title('Truth or Bluff (Polynomial Regression Model) n=2')
plt.show()
```

## Training the polynomial regression model on whole dataset [n=4]
```python
from sklearn.preprocessing  import PolynomialFeatures
poly_fet = PolynomialFeatures(degree=4)
poly_X = poly_fet.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_X,y)
```

## Visualising the Polynomial Regression result
```python
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_X),color='blue')
plt.title('Truth or Bluff (Polynomial Regression Model) n=4')
plt.show()
```

## Visualising the Polynomial Regression result with higher resolution and smoother curve
```python
from sklearn.preprocessing  import PolynomialFeatures

poly_fet = PolynomialFeatures(degree=4)
poly_X = poly_fet.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(poly_X,y)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_fet.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression Model) n=4, smooth curve')
plt.show()
```

## Predicting a new result using Linear Model
```python
lin_reg_1.predict([[6.5]])
```
## Predicting a new result using Polynomial Model
```python
lin_reg_2.predict(poly_fet.fit_transform([[6.5]]))
```

> ## Support Vector Regression [SVR]
In this regression, instead of a regression line, a hyperplane is used.\
Points lying insdie/on the hyperplane are allowed errors, Points lying outside the hyperplane are known as support vector,\
Hence this is known as **Support Vector Regression [SVR]**

### Loading Data and preprocessing it
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Regression/Support_Vector_Regression/Position_Salaries.csv')
# print(dataset)

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

# Taking care of Missing values
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:,:-1])
# X[:,:-1] =  imputer.transform(X[:,:-1])
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X[:,-2:-1])
# X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# print(X)

# Encoding categorial Data [One Hot Encoding]
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# # print(X)

# # Splitting dataset into train and test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_X.fit(X)
X = sc_X.transform(X)

# reshape y
y = y.reshape((len(y),1))

sc_y = StandardScaler()
sc_y.fit(y)
y = sc_y.transform(y)

print(X)
print(y)
```

### Training the SVR model on the whole dataset
```python
from sklearn.svm import SVR
regressor =  SVR(kernel='rbf')
regressor.fit(X,y.ravel())
```

### Predict the salary for new test case
```python
y_pred = regressor.predict(sc_X.transform([[6.5]]))
print(y_pred)
y_pred = sc_y.inverse_transform(y_pred.reshape(1,1))
print(y_pred)
```

### Visualising the SVR model
```python
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(regressor.predict(X).reshape(len(y),1)),color='blue')
plt.title('Truth or Bluff (Support Vector Regression Model)')
plt.show()
```

### Visualising the SVR model in High Resolution
```python
X_grid = np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(X_grid,sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(len(X_grid),1)),color='blue')
plt.title('Truth or Bluff (Support Vector Regression Model) smooth curve')
plt.show()
```

## Decision Tree Regression
In this we add some check [like if else] and based in the condition we predict the value\
**predict value of z, when values of x and y are given**
```python
if x < 20:
    if y < 200:
        z = 300.5
    else:
        z = 65.7
else:
    if y < 170:
        if x < 40:
            z = -64.1
        else:
            z = 0.7
    else:
        z = 1023
```
Basically we split our Dataset graph in various section, and for every new point we find the section in which new data lies, and then for prediction we just take average of that section.

### Load and preprocess the data
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Regression/Decision_Tree_Regression/Position_Salaries.csv')
# print(dataset)

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

# Taking care of Missing values
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:,:-1])
# X[:,:-1] =  imputer.transform(X[:,:-1])
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X[:,-2:-1])
# X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# print(X)

# Encoding categorial Data [One Hot Encoding]
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# # print(X)

# # Splitting dataset into train and test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Training the Decision Tree Regression model on the whole dataset
```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)
```

### Predicting the value for new test using Decision Tree Regressor
```python
regressor.predict([[6.5]])
```

### Visualising the Decision Tree Regression model [High Resolution]
```python
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression Model) smooth curve')
plt.show()
```

## Random Forest Regression [Ensemble Learning]

### Steps/Procedure
1. Pick a number K, and select K random Data Points from the dataset
2. Build a Decision Tree associated with these K points
3. Pick a random[large] number N and build N Decision Tree using Step 1 and Step 2.
4. To predict for a test, First predict using all the N Decision Trees as y<sub>1</sub>, y<sub>2</sub>, y<sub>3</sub>, ... , y<sub>N</sub>.
5. Now take the average of the N values

## Load and preprocess Dataset
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Regression/Random_Forest_Regression/Position_Salaries.csv')
# print(dataset)

X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

# Taking care of Missing values
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:,:-1])
# X[:,:-1] =  imputer.transform(X[:,:-1])
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X[:,-2:-1])
# X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# print(X)

# Encoding categorial Data [One Hot Encoding]
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# # print(X)

# # Splitting dataset into train and test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Train the Random Forest Regression Model using whole dataset
```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X,y)
```

### Predict the result for a new test
```python
regressor.predict([[6.5]])
```

### Visualisation of the Random Forest Regression Model
```python
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Random Forest Regression Model) smooth curve')
plt.show()
```

### Visualisation of the Random Forest Regression Model [High Resolution]
```python
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Random Forest Regression Model) smooth curve')
plt.show()
```

## R-Square method to comare the models
Suppose the points on the regression line are
$$ Y = mx + C $$
$$ RY_{1}, RY_{2}, RY_{3}, ... ,RY_{n}$$
Now lets Assume Y<sub>Avg</sub> is average of all given y points, and the equation for average prediction regression is
$$ Y = Y_{Avg} $$

### Squared Sum of Residuals 
$$ SS_{res} = \sum_{i=1}^{n} (Y_{i}-RY_{i})^2$$

### Squared Sum Total
$$ SS_{tot} = \sum_{i=1}^{n} (Y_{i}-Y_{Avg})^2$$

### R-Square
$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
**If value of  R<sup>2</sup> is nearer to 1, model is good.**

## Adjusted R<sup>2</sup>
It is observed that, whenever you add a new independent variable in regression equation, SS<sub>res</sub> is either going to increase or be the same, So R<sup>2</sup> will always increase, hence this method will not help in case of adding a new variable in regression, Now
$$ R^2_{adj} = 1 - (1-R^2) \frac{n-1}{n-p-1}$$
n - Sample Size<br/> 
p - No of Independent variable used in regression

<br/>
<hr/>
<br/>

> ## Regression Model Selection [ Comparing and finding best regresssion Model]
### Load and preprocess Dataset
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Regression/Model_Selection_Regression/Data.csv')
# print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(X)
print(y)

# Taking care of Missing values
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:,:-1])
# X[:,:-1] =  imputer.transform(X[:,:-1])
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X[:,-2:-1])
# X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# print(X)

# Encoding categorial Data [One Hot Encoding]
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# # print(X)

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

### Prediction using Multiple Linear Regression
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_MLR = regressor.predict(X_test)
print(y_pred_MLR)
```

### Prediction using Polynomial Regression
```python
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_fet = PolynomialFeatures(degree=4)
poly_X = poly_fet.fit_transform(X_train)
poly_reg =  LinearRegression()
poly_reg.fit(poly_X,y_train)
y_pred_PR = poly_reg.predict(poly_fet.fit_transform(X_test))
print(y_pred_PR)
```

### Prediction using Support Vector Regression
```python
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_X.fit(X_train)
X_sc = sc_X.transform(X_train)

# reshape y
y_sc = y_train.reshape((len(y_train),1))

sc_y = StandardScaler()
sc_y.fit(y_sc)
y_sc = sc_y.transform(y_sc)

# print(X_sc)
# print(y_sc)

from sklearn.svm import SVR
regressor =  SVR(kernel='rbf')
regressor.fit(X_sc,y_sc.ravel())

y_pred_SVR = regressor.predict(sc_X.transform(X_test))
print(y_pred_SVR)
y_pred_SVR = sc_y.inverse_transform(y_pred_SVR.reshape(len(y_pred_SVR),1)).ravel()
print(y_pred_SVR)
```

### Prediction using Decision Tree Regression
```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
y_pred_DTR = regressor.predict(X_test)
print(y_pred_DTR)
```

### Prediction using Random Forest Regression
```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train,y_train)
y_pred_RFR = regressor.predict(X_test)
print(y_pred_RFR)
```

## R-Square Comparision for All model
```python
from sklearn.metrics import r2_score
R2S = {}
R2S['Multiple Linear Regression'] = r2_score(y_test, y_pred_MLR)
R2S['Polynomial Regression'] = r2_score(y_test, y_pred_PR)
R2S['Support Vector Regression'] = r2_score(y_test, y_pred_SVR)
R2S['Decision Tree Regression'] = r2_score(y_test, y_pred_DTR)
R2S['Random Forest Regression'] = r2_score(y_test, y_pred_RFR)

for method in R2S:
    print(method,":",R2S[method])
```

<hr/>
<hr/>

> # **Logistic Regression**

Instead of predicting a value, when we try to pridict probability of any event, We use Logistic regression. The outcome of this forecast lies between 0 and 1

$$ y = b_{0} + b_{1}x $$

If we pass above linear eqaution to below Sigmoid function

$$ p = \frac{1}{1 + e^{-y}} $$

The generated eqation by comparing y on both equation comes out

$$ \ln{(\frac{p}{1 - p})} =  b_{0} + b_{1}x  $$

This above eqation is used in **Logistic Regression**.

Now we select a threshold value [0,1], if predicted probability is less than the threshhold value, Outcome is NO else YES.

## Loading and Preprocession Data
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Classification/Logistic_Regression/Social_Network_Ads.csv')
# print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# # Taking care of Missing values
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:,:-1])
# X[:,:-1] =  imputer.transform(X[:,:-1])
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X[:,-2:-1])
# X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# # print(X)

# # Encoding categorial Data [One Hot Encoding]
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# # print(X)

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
```

## Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# fit using training data
sc.fit(X_train)
# transform training data using the scaler
X_train = sc.transform(X_train)
# transform test data using the same scaler
X_test = sc.transform(X_test)
```

## Training the Logistic Regression Model using Training Data
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)
```

## Predicting a test resullt
```python
# predict a single test 
classifier.predict(sc.transform([[30,87000]]))
```

## Predict Test results
```python
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),axis=1))
```

## Making the Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
acs = accuracy_score(y_test,y_pred)
print(acs)
# Confusion Matrix
# [
#     [Correct-0,   Incorrect-1]
#     [Incorrect-0, Correct-0]
# ]
```

## Visualising the Training set results
```python
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## Visualising the Test set results
```python
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
<hr/>
<hr/>

> # K-Nearest Neighbour

In this type of classification technique, we first select a value for K and then follow the below steps

1. Select the value of K. Generally the value of K is choosen is 5.
2. Select K Nearest Neighbour of the new Data Point. Usually Eucledian Distance is used to select Nearest Neighbours.
$$ D_{Eucledian\ Distance} = \sqrt{(x_{2}-x_{1})^{2}+(y_{2}-y_{1})^{2}}$$
3. Among these K Neighbours, count the number of datapoints in each category.
4. Assign the new data point to the category with most counted nearest neighbour.
5. Model is ready.


## Loading and Preprocession Data
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('../Datasets/Classification/K_Nearest_Neighbour/Social_Network_Ads.csv')
# print(dataset)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# # Taking care of Missing values
# from sklearn.impute import SimpleImputer 
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:,:-1])
# X[:,:-1] =  imputer.transform(X[:,:-1])
# imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imputer.fit(X[:,-2:-1])
# X[:,-2:-1] =  imputer.transform(X[:,-2:-1])
# # print(X)

# # Encoding categorial Data [One Hot Encoding]
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[-1])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))
# # print(X)

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
```

## Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# fit using training data
sc.fit(X_train)
# transform training data using the scaler
X_train = sc.transform(X_train)
# transform test data using the same scaler
X_test = sc.transform(X_test)
```

## Training the K Nearest Neighbour Model using Training Data
```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier.fit(X_train,y_train)
```

## Predicting a test resullt
```python
# predict a single test 
classifier.predict(sc.transform([[30,87000]]))
```

## Predict Test results
```python
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),axis=1))
```

## Making the Confusion Matrix
```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
acs = accuracy_score(y_test,y_pred)
print(acs)
# Confusion Matrix
# [
#     [Correct-0,   Incorrect-1]
#     [Incorrect-0, Correct-0]
# ]
```

## Visualising the Training set results
```python
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 5),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 5))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K Nearest Neighbour (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## Visualising the Test set results
```python
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 5),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 5))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K Nearest Neighbour (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```