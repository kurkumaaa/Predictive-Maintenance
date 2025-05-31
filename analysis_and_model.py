import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def analysis_and_model_page():
    st.title("Анализ данных и модель")
    
    # Загрузка данных
    uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Данные успешно загружены!")
        st.write("Первые 5 строк данных:", data.head())
        
        # Предобработка данных
        st.header("Предобработка данных")
        
        # Удаление ненужных столбцов
        cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        cols_to_drop = [col for col in cols_to_drop if col in data.columns]
        data = data.drop(columns=cols_to_drop)
        
        # Преобразование категориальных переменных
        if 'Type' in data.columns:
            le = LabelEncoder()
            data['Type'] = le.fit_transform(data['Type'])
        
        # Проверка на пропущенные значения
        st.write("Пропущенные значения:", data.isnull().sum())
        
        # Масштабирование числовых признаков
        numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        numerical_features = [col for col in numerical_features if col in data.columns]
        
        if numerical_features:
            scaler = StandardScaler()
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
        
        # Разделение данных
        st.header("Разделение данных")
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Обучающая выборка: {X_train.shape[0]} записей")
        st.write(f"Тестовая выборка: {X_test.shape[0]} записей")
        
        # Обучение модели
        st.header("Обучение модели")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Оценка модели
        st.header("Оценка модели")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"ROC-AUC: {roc_auc:.4f}")
        
        # Визуализация результатов
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Предсказанные')
        ax.set_ylabel('Фактические')
        st.pyplot(fig)
        
        st.subheader("Classification Report")
        st.text(class_report)
        
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)
        
        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            st.write("Введите значения признаков:")
            
            col1, col2 = st.columns(2)
            with col1:
                type_val = st.selectbox("Тип оборудования (L=0, M=1, H=2)", [0, 1, 2])
                air_temp = st.number_input("Температура воздуха [K]", value=300.0)
                process_temp = st.number_input("Температура процесса [K]", value=310.0)
            with col2:
                rotational_speed = st.number_input("Скорость вращения [rpm]", value=1500)
                torque = st.number_input("Крутящий момент [Nm]", value=40.0)
                tool_wear = st.number_input("Износ инструмента [min]", value=100)
            
            submit_button = st.form_submit_button("Предсказать")
            
            if submit_button:
                # Создание DataFrame с введенными данными
                input_data = pd.DataFrame({
                    'Type': [type_val],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear]
                })
                
                # Масштабирование (используем тот же scaler)
                if numerical_features:
                    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
                
                # Предсказание
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]
                
                st.success(f"Результат предсказания: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                st.info(f"Вероятность отказа: {prediction_proba[0]:.4f}")

# Запуск страницы
if __name__ == "__main__":
    analysis_and_model_page()