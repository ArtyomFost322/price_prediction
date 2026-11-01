import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(13)
n = 200

area = np.random.randint(30, 120, n)
distance = np.random.randint(1, 20, n)
base_price = (area * 3500) - (distance * 8000)
market_noise = np.random.normal(20000, 12000, n) 
price = base_price + market_noise

df = pd.DataFrame({
    'Area_Sqm': area, 
    'Distance_Km': distance, 
    'Price_USD': price
})
print(df.head(), "\n")

plt.style.use('seaborn-v0_8-whitegrid')
graphite_color = '#2C3E50'
accent_orange = '#E67E22'

plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='cividis', fmt=".2f", cbar=False)
plt.title('Features Correlation', fontsize=11, pad=12, color=graphite_color)
plt.tight_layout()
plt.savefig('corr_matrix.png', dpi=150)
plt.close()

X = df[['Area_Sqm', 'Distance_Km']]
y = df['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"R2: {r2_score(y_test, preds):.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, color=graphite_color, alpha=0.5, edgecolors='none', s=45)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color=accent_orange, linestyle='--', lw=2)

plt.xlabel('Actual Price ($)', color=graphite_color)
plt.ylabel('Predicted Price ($)', color=graphite_color)
plt.title('Model Predictions vs Reality', fontsize=12, pad=15, color=graphite_color)
plt.tight_layout()
plt.savefig('preds_vs_actual.png', dpi=150)
plt.close()

