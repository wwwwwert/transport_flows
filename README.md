# Transport Flows Analysis

Проект для анализа задач P-median на транспортных сетях с временными данными о скорости движения.

## Структура проекта

### Корневая директория

- `requirements.txt` - зависимости Python
- `.gitignore` - файлы для игнорирования в git
- `yandex_maps.ipynb` - работа с API Яндекс.Карт

### compare_distances/

Основной модуль для анализа P-median задач с различными метриками расстояний.

#### compare_distances/scripts/

Основные модули проекта:

- `__init__.py` - инициализация пакета, экспорт всех функций
- `graph_operations.py` - операции с графами, загрузка данных, вычисление расстояний
- `optimization.py` - решение задач P-median (точные и эвристические алгоритмы)
- `analysis.py` - анализ результатов решения
- `visualization.py` - визуализация графов и результатов

#### compare_distances/converted_notebooks/

Готовые скрипты для анализа:

- `solve_p_median.py` - основной скрипт для решения P-median задач
- `compare_p_median.py` - сравнительный анализ покрытия для разных количеств объектов

#### compare_distances/*.ipynb

Jupyter ноутбуки для разработки и экспериментов:

- `solve_p_median.ipynb` - интерактивная версия основного анализа
- `compare_p_median.ipynb` - интерактивная версия сравнительного анализа
- `P-median distances Tata_upd_refactored.ipynb` - рефакторенная версия анализа
- `P-median distances Tata_upd_coverage_refactored.ipynb` - анализ покрытия

### chikago_graph/

Анализ данных о трафике в Чикаго:

- `chikago_graph.ipynb` - основной анализ графа Чикаго
- `create_chikago_dataset.ipynb` - создание датасета из исходных данных

### new_york_graph/

Анализ данных о трафике в Нью-Йорке:

- `dot_nyc_graph.ipynb` - анализ данных NYC DOT
- `create_dot_nyc_dataset.ipynb` - подготовка данных NYC

### urban_traffic_graph/

Анализ городского трафика:

- `urban_traffic.ipynb` - основной анализ городского трафика
- `test_graph.ipynb` - тестирование алгоритмов на графах
- `draw_edgelist.ipynb` - визуализация графов из edgelist файлов
- `road_volume_series.csv` - данные об объемах трафика

### ivan_scripts/

Оригинальные скрипты:

- `P-median distances Tata_upd.ipynb` - оригинальная версия анализа
- `P-median distances Tata_upd_coverage.ipynb` - оригинальный анализ покрытия
- `Tata_upd.csv` - тестовые данные
- `GUF_belgrade.pdf` - презентация по алгоритму GUF

<!-- ## Основные модули

### graph_operations.py

Функции для работы с графами:

- `load_graph_from_csv()` - загрузка графа из CSV файла
- `create_graphs()` - создание igraph и networkx представлений
- `create_graphs_from_edgelist()` - загрузка из edgelist формата
- `calculate_distance_matrices()` - вычисление матриц расстояний
- `protected_distance()` - вычисление защищенных расстояний
- `protected_distance_speed_graph()` - защищенные расстояния с учетом скорости
- `has_speed_attributes()` - проверка наличия атрибутов скорости
- `prune_leaf_nodes()` - удаление листовых узлов

### optimization.py

Алгоритмы оптимизации:

- `solve_pmedian_problem()` - основная функция решения P-median
- `solve_pmedian_teitz_bart()` - эвристический алгоритм Teitz-Bart
- `create_pmedian_model()` - создание модели оптимизации в Pyomo
- `extract_solution_results()` - извлечение результатов решения

### analysis.py

Анализ результатов:

- `extract_distances_from_solution()` - извлечение расстояний из решения
- `create_results_dataframe()` - создание DataFrame с результатами

### visualization.py

Визуализация:

- `draw_graph_with_centers()` - отрисовка графа с выделенными центрами
- `draw_networkx_auto()` - автоматический выбор способа визуализации
- `draw_graph_on_map()` - отрисовка на географической карте
- `plot_multiple_quantile_distributions()` - графики квантильных распределений
- `extract_node_coordinates_from_edges()` - извлечение координат из атрибутов ребер -->

<!-- ## Основные скрипты

### solve_p_median.py

Основной скрипт для решения P-median задач. Выполняет:

1. Загрузку графа из edgelist файла
2. Вычисление трех типов расстояний (геодезические, сопротивления, защищенные)
3. Решение P-median задач для каждого типа расстояний
4. Визуализацию результатов
5. Статистический анализ решений
6. Сохранение всех результатов в структурированном виде

Поддерживает три типа весов ребер:
- `binary` - единичные веса
- `length` - веса по длине ребер
- `time` - веса по времени прохождения (длина/скорость)

### compare_p_median.py

Скрипт для сравнительного анализа покрытия. Выполняет:

1. Решение P-median задач для разного количества объектов (p=1..9)
2. Анализ наихудших расстояний для каждого решения
3. Сравнение трех типов расстояний
4. Построение графиков сравнения
5. Сохранение результатов анализа -->

## Форматы данных

### Edgelist формат
```
node1 node2 {'length': 100.0, 'speed_08:00:00': 25.5, 'speed_08:05:00': 23.2, ...}
```

### CSV формат
```csv
source,target,length,category
1,2,150.0,highway
2,3,200.0,street
```

## Типы расстояний

1. **Геодезические расстояния** - стандартные кратчайшие пути
2. **Расстояния сопротивления** - на основе теории электрических сетей
3. **Защищенные расстояния** - максимальные кратчайшие пути после удаления любого ребра
4. **Защищенные расстояния со скоростью** - максимальное время в пути по всем временным интервалам

## Получение результатов

Для получения графиков результатов можно запустить ноутбуки `solve_p_median.ipynb` и `compare_p_median.ipynb`

Для получения результатов с точными солверами можно запусить скрипты `converted_notebooks/solve_p_median.py` и `converted_notebooks/compare_p_median.py`

Результаты автоматически сохраняются в структурированном виде:

```
data/results/
└── graph_name/
    └── weight_type/
        ├── original_graph.png
        ├── pmedian_solutions_comparison.png
        ├── quantile_distributions.png
        ├── analysis_summary.txt
        ├── statistical_summary.csv
        └── results_data.csv
```

<!-- ## Зависимости

Основные библиотеки:
- `networkx` - анализ графов
- `igraph` - высокопроизводительные операции с графами
- `pandas` - обработка данных
- `numpy` - численные вычисления
- `pyomo` - моделирование оптимизации
- `matplotlib`, `seaborn` - визуализация
- `contextily`, `geopandas` - географическая визуализация

Солверы оптимизации:
- `scip` - рекомендуемый точный солвер
- `glpk` - альтернативный точный солвер -->
