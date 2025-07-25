# Проект: Бинарная классификация для предиктивного обслуживания оборудования

![Predictive Maintenance](https://img.icons8.com/color/96/000000/maintenance.png)

## Описание проекта

Цель проекта — разработать модель машинного обучения, которая предсказывает вероятность отказа промышленного оборудования на основе данных датчиков. Модель классифицирует состояние оборудования на:
- **0**: Нормальная работа
- **1**: Вероятный отказ

Реализовано многостраничное Streamlit-приложение с:
- Анализом данных и прогнозированием
- Интерактивной презентацией проекта

## Датасет

Используется **[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)**:

- **10,000 записей** с параметрами оборудования
- **14 признаков**, включая:
  - Температурные показатели (воздух/процесс)
  - Параметры работы (скорость вращения, крутящий момент)
  - Показатели износа инструмента
- **5 типов отказов** (TWF, HDF, PWF, OSF, RNF)

Пример данных:
| UID | Type | Air Temp [K] | Rotational Speed [rpm] | Machine failure |
|-----|------|--------------|------------------------|-----------------|
| 1   | L    | 298.1        | 1551                   | 0               |
| 2   | M    | 299.5        | 1428                   | 1               |

## Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone https://github.com/kurkumaaa/Predictive-Maintenance
cd Predictive-Maintenance

Установите зависимости:
pip install -r requirements.txt
Запустите приложение:
streamlit run app.py
