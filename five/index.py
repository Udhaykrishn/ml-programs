import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1000,2],[1500,3],[1200,2],[1800,4],[900,2],[2000,3]])
y = np.array([300000,400000,350000,500000,280000,450000])

model = LinearRegression()

model.fit(X,y)

size = float(input("Enter the Size of the housr if sqrt: "))
bedrooms = int(input("Enter the number of bedrooms: "))
new_data = np.array([[size,bedrooms]])

pre_price = model.predict(new_data)

print("Predicted price for a house with size {} sqrt and {} bedrooms: Rs.{:.2f}".format(size,bedrooms,pre_price[0]))