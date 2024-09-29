import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('code.csv')
print(data.head())
print(data.describe())
X = data[["CUSTOMER_REVIEW","CELEBRITY_REACH","COMPETETION","INNOVATION","QUALITY","TRANSPARENCY","COMMUNICATION","TARGET_GROUP","FREQUENT_REMINDERS"]]
Y = data["IMPACT_RATE"]
Y_ = data["DESCISION_MAKING"]
plt.scatter(X['CUSTOMER_REVIEW'], Y, color='b')
plt.xlabel('CUSTOMER_REVIEW')  
plt.ylabel('IMPACT_RATE') 
plt.show()
inps = [[8,2,79,7,5,3,3,56,2]]
mdl = RandomForestRegressor(n_estimators=100,max_depth=6)
mdl.fit(X, Y)
pred = mdl.predict(inps)
print("Predicted value (RFR): ",pred[0])
print("Accuracy (RFR): ",mdl.score(X[:100], Y[:100])*100)
plt.scatter(X['CUSTOMER_REVIEW'], Y, color='b')
plt.plot(X['CUSTOMER_REVIEW'], mdl.predict(X),color='cyan',linewidth=3)
plt.xlabel('CUSTOMER_REVIEW')  
plt.ylabel('High/Low') 
plt.show()
