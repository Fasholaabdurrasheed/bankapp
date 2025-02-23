import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

commerce = pd.read_csv('Test.csv')
commerce['Sales_Quantity'] = [random.randint(1, 10) for i in range(len(commerce['Sales_Quantity']))]
group_file = commerce.groupby('Category')['Price'].mean()
print(group_file)
x = commerce[['Price']]
y = commerce['Sales_Quantity']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)
## Mke predicion
y_pred = model.predict(X_test)
## Evaluate the model
mea = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mea}")
plt.scatter(y_test, y_pred, marker='*', color= 'blue')
plt.grid()
plt.xlabel('Category')
plt.ylabel('Price')
plt.title('Trending Goods')
plt.legend(group_file)
plt.show()
sys.stdout.flush()


