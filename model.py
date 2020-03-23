import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/prepared_data.csv', sep=';')

# разделим признаки на числовые и категориальные
num_cols = ['Course_averageScore',
            'Course_passingScore',
            'Test_Qnt',
            'Test_durationAvg',
            'Student_MessagesQnt']

obj_cols = ['Course_visibilityStatus',
            'Course_periodicity']

# Кодируем категориальные признаки
ohe_df = pd.DataFrame(index=df['student_id'])
ohe = OneHotEncoder(handle_unknown='ignore')

for col in obj_cols:
    ohe.fit(df[[col]])

    ohe_result = pd.DataFrame(ohe.transform(df[[col]]).toarray(),
                              columns=ohe.get_feature_names(input_features=[col]),
                              index=df['student_id'])
    ohe_df = ohe_df.join(ohe_result)

# Стандартизируем числовые признаки
std_df = pd.DataFrame(index=df['student_id'])
scaler = StandardScaler()

for col in num_cols:
    scaler.fit(df[[col]])
    std_result = pd.DataFrame(scaler.transform(df[[col]]),
                              columns=[col],
                              index=df['student_id'])
    std_df = std_df.join(std_result, on='student_id', how='left', lsuffix='_left', rsuffix='_right')

    X = ohe_df.join(std_df)
y = df['Churn']

# Отделяем тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, stratify=y, random_state=18)

# Обучаем логистическую регрессию
lr = LogisticRegression(C=1.0, random_state=18, n_jobs=-1).fit(X_train, y_train)

# Делаем прогноз и оцениваем результат
y_pred_test = lr.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, y_pred_test)

# Для примера предсказываем значения для всей выборки
y_pred = lr.predict_proba(X)[:, 1]
results = list(zip(df['student_id'], y_pred, y))
results_df = pd.DataFrame(results, columns=['student_id', 'prob_predicted', 'Real_value'])
results_df['Predicted_value'] = results_df['prob_predicted'].apply(lambda x: 1 if x >= 0.81 else 0)

# Сохраняем результат
df.to_csv('data/churn_prediction.csv', index=False, sep=';')
