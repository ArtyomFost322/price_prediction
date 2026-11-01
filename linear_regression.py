# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score

# # 1. ГЕНЕРАЦИЯ ДАННЫХ (Симуляция рынка недвижимости)
# np.random.seed(42)
# n_samples = 200

# # Генерируем площадь (от 30 до 120 кв.м.)
# area = np.random.randint(30, 120, n_samples)
# # Генерируем расстояние до центра (от 1 до 20 км)
# distance = np.random.randint(1, 20, n_samples)
# # Формула цены с добавлением случайного шума (рыночных колебаний)
# price = (area * 800) - (distance * 1200) + np.random.normal(30000, 5000, n_samples)

# # Собираем все в таблицу DataFrame
# data = pd.DataFrame({'Area_Sqm': area, 'Distance_Km': distance, 'Price_USD': price})

# print("--- Первые 5 строк сгенерированного датасета ---")
# print(data.head(), "\n")

# # 2. АНАЛИЗ ДАННЫХ (Построение матрицы корреляции)
# plt.figure(figsize=(6, 4))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Матрица корреляции признаков')
# plt.tight_layout()
# plt.savefig('correlation_matrix.png') # Сохраняем график для GitHub
# plt.close()

# # 3. ПОДГОТОВКА И ОБУЧЕНИЕ МОДЕЛИ
# X = data[['Area_Sqm', 'Distance_Km']] # Факторы
# y = data['Price_USD']                  # Целевая переменная (цена)

# # Разделяем на обучающую выборку (80%) и тестовую (20%)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Инициализируем и обучаем модель Линейной Регрессии
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 4. ОЦЕНКА РЕЗУЛЬТАТОВ
# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("--- Метрики эффективности модели ---")
# print(f"Средняя ошибка предсказания (MAE): ${mae:.2f}")
# print(f"Коэффициент детерминации (R² балл): {r2:.4f} (Чем ближе к 1, тем лучше)\n")

# # 5. ВИЗУАЛИЗАЦИЯ ИТОГОВ (Реальные цены vs Предсказанные)
# plt.figure(figsize=(6, 6))
# plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Предсказания')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Идеальный прогноз')
# plt.xlabel('Реальная цена ($)')
# plt.ylabel('Предсказанная цена ($)')
# plt.title('Сравнение реальных и предсказанных цен')
# plt.legend()
# plt.tight_layout()
# plt.savefig('predictions_vs_reality.png') # Сохраняем второй график
# print("Проект успешно выполнен! Графики сохранены в папку с кодом.")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Генерируем случайные данные для симуляции рынка
np.random.seed(42)
n_samples = 200

area = np.random.randint(30, 120, n_samples)          # площадь от 30 до 120 кв.м.
distance = np.random.randint(1, 20, n_samples)      # расстояние до центра от 1 до 20 км

# Считаем цену с добавлением случайного шума
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(13) # нестандартный сид выглядит естественнее
n = 200

# Синтетика для теста
area = np.random.randint(30, 120, n)
distance = np.random.randint(1, 20, n)
# Базовая формула цены + шум
price = (area * 800) - (distance * 1200) + np.random.normal(30000, 5000, n)

df = pd.DataFrame({
    'Area_Sqm': area, 
    'Distance_Km': distance, 
    'Price_USD': price
})
print(df.head())

# Эда / Корреляции
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('corr_matrix.png')
plt.close()

# Моделирование
X = df[['Area_Sqm', 'Distance_Km']]
y = df['Price_USD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

lr = LinearRegression()
lr.fit(X_train, y_train)

preds = lr.predict(X_test)

# Валидация
print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
print(f"R2: {r2_score(y_test, preds):.4f}")

# График остатков / предсказаний
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.savefig('preds_vs_actual.png')
plt.close()
