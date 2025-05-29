import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Исходные данные (процент Eu и соответствующий Eg)
X_train = np.array([[1], [5], [10], [20]])
y_train = np.array([4.77, 4.54, 4.31, 4.17])

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Интерфейс Streamlit
st.set_page_config(page_title="Eg предсказание", layout="centered")
st.title("💡 Предсказание ширины запрещённой зоны Eg")
st.subheader("на основе содержания Eu в Ca₁₋ₓEuₓWO₄")

st.markdown("""
Это мини-программа использует модель машинного обучения для предсказания ширины запрещённой зоны (Eg) на основе доли замещения кальция на европий в кристалле шерлита CaWO₄.
""")

# Слайдер для ввода процента Eu
eu_percent = st.slider("Процент замещения кальция (Eu³⁺), %", min_value=0.0, max_value=25.0, value=5.0, step=0.1)

# Предсказание
predicted_Eg = model.predict([[eu_percent]])[0]
st.success(f"🔍 Предсказанное значение Eg: **{predicted_Eg:.2f} эВ**")

# Визуализация
x_vals = np.linspace(0, 25, 200).reshape(-1, 1)
y_vals = model.predict(x_vals)

fig, ax = plt.subplots()
ax.plot(X_train, y_train, 'o', label='Экспериментальные данные')
ax.plot(x_vals, y_vals, '-', label='ML модель')
ax.axvline(eu_percent, color='red', linestyle='--', label='Точка предсказания')
ax.axhline(predicted_Eg, color='green', linestyle='--')
ax.set_xlabel("Содержание Eu³⁺, %")
ax.set_ylabel("Eg (эВ)")
ax.set_title("Зависимость Eg от содержания Eu")
ax.legend()
st.pyplot(fig)

st.caption("🔬 Данные: Ca₁₋ₓEuₓWO₄, спектроскопия в ИК-области, 2025")