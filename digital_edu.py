import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer


def fill_ed(education_status):
    if education_status == 'Undergraduate applicant':
        return 1
    elif education_status == 'Student (Bachelor\'s)':
        return 2
    elif education_status == 'Alumnus (Bachelor\'s)':
        return 3
    elif education_status == 'Student (Master\'s)':
        return 4
    elif education_status == 'Alumnus (Master\'s)':
        return 5
    elif education_status == 'Candidate of Sciences':
        return 6
    return 0

def fill_sex(sex):
    if sex == 2:
        return 1
    return 0

# Шаг 1. Загрузка и очистка данных
df = pd.read_csv('train.csv')

# Удаление ненужных столбцов
df.drop(['id', 'life_main', 'people_main', 'bdate', 'education_form', 
         'langs', 'city', 'last_seen', 'occupation_type', 'occupation_name',
           'career_start', 'career_end'], axis=1, inplace=True)

# Преобразование категориальных признаков в числовые
df['sex'] = df['sex'].apply(fill_sex) # замена пола на бинарные значения (1 - женский, 0 - мужской)

df['relation'] = df['relation'].fillna(0) # заполнение пропущенных значений в семейном положении


df['education_status'] = df['education_status'].apply(fill_ed)
df['education_status'].fillna(0)



# Удаление строк с оставшимися пропущенными значениями
df.dropna(inplace=True)
print(df.info())