import pandas as pd
import numpy as np

df = pd.read_csv('data/raw_data.csv', sep=';')

df['user_regDate'] = pd.to_datetime(df['user_regDate'], dayfirst=True)

# NaN to 0
zero_cols = [  # Course
    'Course_publication_updateDiff', 'Course_publicationDateYear', 'Course_publicationDateMonth', 'Course_description',
    'Course_DurationDays', 'Course_LessonTestQnt', 'Course_LessonTheoryQnt', 'Course_LessonMediaQnt',
    'Course_passingScore', 'Course_averageScore'
    # Student
    'Student_lastVisitedDiff', 'Student_name', 'Student_secondName', 'Student_MessagesQnt',
    # CourseOwner
    'CourseOwner_lastVisitedDiff',
    # Test
    'Test_durationAvg', 'Test_durationMax', 'Test_durationMin']

df[zero_cols] = df[zero_cols].fillna(0)

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

# Сохраняем результат
df.to_csv('data/prepared_data.csv', index=False, sep=';')
