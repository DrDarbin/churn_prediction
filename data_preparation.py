import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/raw_data.csv', sep=';')

df['user_regDate'] = pd.to_datetime(df['user_regDate'], dayfirst=True)

# NaN to 0
# Удалены переменные: 'Course_description', 'Student_name', 'Student_secondName'
zero_cols = [  # Course
    'Course_publication_updateDiff', 'Course_publicationDateYear', 'Course_publicationDateMonth', 
    'Course_DurationDays', 'Course_LessonTestQnt', 'Course_LessonTheoryQnt', 'Course_LessonMediaQnt',
    'Course_passingScore', 'Course_averageScore',
    # Student
    'Student_lastVisitedDiff', 'Student_MessagesQnt',
    # CourseOwner
    'CourseOwner_lastVisitedDiff',
    # Test
    'Test_durationAvg', 'Test_durationMax', 'Test_durationMin']

# New features based on zero_cols (True for NaN)
for col in zero_cols:
    df[col + '_was_missing'] = df[col].isnull()

# Nan to mean (instead of zeros)
df_copy = df[zero_cols].copy()

imputer = SimpleImputer()
df_imputed = pd.DataFrame( imputer.fit_transform(df_copy) )
df_imputed.columns = df_copy.columns

df[zero_cols] = df_imputed[zero_cols]

# date columns
date_cols = ['Course_startAt', 'Course_stopAt']
df[date_cols] = df[date_cols].astype('datetime64[D]')

# int columns
int_cols = [  # Course
    'Course_publication_updateDiff', 'Course_publicationDateYear', 'Course_publicationDateMonth', 'Course_cost',
    'Course_DurationDays', 'Course_LessonTestQnt', 'Course_LessonTheoryQnt', 'Course_LessonMediaQnt',
    'Course_studentsQnt', 'Course_isDeleted',
    # Student
    'Student_id', 'Student_lastVisitedDiff', 'Student_MessagesQnt', 'Student_RegYear', 'Student_RegMonth',
    'Student_RegDay', 'Student_RegDoW', 'Student_RegDiff', 'Student_lastActivityDiff', 'Student_isDeleted',
    # CourseOwner
    'CourseOwner_lastVisitedDiff', 'CourseOwner_RegMonth', 'CourseOwner_RegDiff', 'CourseOwner_RegYear',
    # Test
    'Test_Qnt', 'Test_durationAvg', 'Test_durationMax', 'Test_durationMin']

df[int_cols] = df[int_cols].astype(int)

# float columns
float_cols = ['Course_passingScore', 'Course_averageScore', 'Student_degreeCourse']

df[float_cols] = df[float_cols].astype(float)

# object columns
obj_cols = ['Student_status', 'Course_visibilityStatus', 'Course_periodicity']

df[obj_cols] = df[obj_cols].astype('O')

# isFree
df['Course_isFree'] = df['Course_cost'].apply(lambda x: 1 if x == 0 else 0)

# оставляем только активных студентов и курсы
df = df[(df['Student_isDeleted'] == 0) & (df['Course_isDeleted'] == 0)]

df['Churn'] = np.where((df['Student_status'] == 'no_active') & (df['Student_degreeCourse'] < 0.95), 1, 0)

df = df[(df['Course_studentsQnt'] > 1)]

# Drop outliers for numeric columns (Tukey method)
for col in list(np.append(int_cols, float_cols)):
    Q1 = np.percentile(df[col], 25)
    Q3 = np.percentile(df[col],75)
    
    # Interquartile range
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
        
    # Outliers indices
    outliers_indicies = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
    df.drop(outliers_indicies, axis = 0)

# New features:
df['Course_difficulty'] = df['Course_averageScore']/df['Course_passingScore']

# разделим признаки на числовые и категориальные
num_cols = ['Course_averageScore',
            'Course_passingScore',
            'Test_Qnt',
            'Test_durationAvg',
            'Student_MessagesQnt',
            
            # additional probably important features
            'Student_RegDiff',
            'Student_RegMonth',
            'Student_lastVisitedDiff',
            'Course_DurationDays',
            'Course_LessonTestQnt', 
            'Course_LessonTheoryQnt', 
            'Course_LessonMediaQnt',
            'Course_cost',
            'Course_isFree',
            'Course_studentsQnt',
            'CourseOwner_lastVisitedDiff',
            
            # new generated features
            'Course_averageScore_was_missing',
            'Course_passingScore_was_missing',
            'Test_durationAvg_was_missing',
            'Student_MessagesQnt_was_missing',
            'Student_lastVisitedDiff_was_missing',
            'Course_DurationDays_was_missing',
            'Course_LessonTheoryQnt_was_missing', 
            'Course_LessonMediaQnt_was_missing',
            'Course_passingScore_was_missing',
            'CourseOwner_lastVisitedDiff_was_missing',
            'Course_difficulty']

cat_cols = ['Course_visibilityStatus',
            'Course_periodicity']

# Кодируем категориальные признаки
ohe_df = pd.DataFrame(index=df['Student_id'])
ohe = OneHotEncoder(handle_unknown='ignore')

for col in cat_cols:
    ohe.fit(df[[col]])
    ohe_result = pd.DataFrame(ohe.transform(df[[col]]).toarray(),
                              columns=ohe.get_feature_names(input_features=[col]),
                              index=df['Student_id'])
    ohe_df = ohe_df.join(ohe_result)

# Стандартизируем числовые признаки
std_df = pd.DataFrame(index=df['Student_id'])
scaler = StandardScaler()

for col in num_cols:
    scaler.fit(df[[col]])
    std_result = pd.DataFrame(scaler.transform(df[[col]]),
                              columns=[col],
                              index=df['Student_id'])
    std_df = std_df.join(std_result, on='Student_id', how='left', lsuffix='_left', rsuffix='_right')

df_prepared = ohe_df.join(std_df).join(df['Churn'])

# Сохраняем результат
df_prepared.to_csv('data/prepared_data.csv', index=True, sep=';')