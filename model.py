import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

df = pd.read_csv('data/prepared_data.csv', sep=';')

y = df['Churn']
X = df.drop(['Churn'], axis=1)

# Обучаем логистическую регрессию на Kfold вместо train_test_split
# Auto selecting the best Regularization value (Cs)
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=33)
c_values = np.logspace(-2, 3, 100)
lr = LogisticRegressionCV(Cs=c_values, cv=skf, verbose=0, n_jobs=-1, scoring="roc_auc", max_iter=1000)
lr.fit(X.values, y.values)

# The best regularization parameter Cs (L2) and ROC AUC Score
Cs = lr.C_
score = lr.score(X, y)
print('The best L2: ', Cs, '\n\r', 'ROC AUC: ', score)

# Для примера предсказываем значения для всей выборки
y_pred = lr.predict_proba(X)[:, 1]
results = list(zip(df['Student_id'], y_pred, y))
results_df = pd.DataFrame(results, columns=['Student_id', 'prob_predicted', 'Real_value'])
results_df['Predicted_value'] = results_df['prob_predicted'].apply(lambda x: 1 if x >= 0.81 else 0)

# Сохраняем результат
results_df.to_csv('data/churn_prediction.csv', index=False, sep=';')