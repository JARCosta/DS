

import array
from numpy import ndarray
from pandas import read_csv, concat, DataFrame, Series
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def open(file_tag: str) -> DataFrame:
    data: DataFrame = read_csv(f"data/{file_tag}.csv", sep=",", decimal=".")
    return data

def smotify(file_tag: str, target: str):

    original: DataFrame = open(file_tag)

    target_count: Series = original[target].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()


    RANDOM_STATE = 42

    smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
    y = original.pop(target).values
    X: ndarray = original.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(original.columns) + [target]
    df_smote.to_csv(f"data/{file_tag}_smote.csv", index=False)
    
    return df_smote

def split(file_tag: str, target: str):

    data : DataFrame = open(file_tag)

    y: array = data.pop(target).to_list()
    X: ndarray = data.values


    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    train: DataFrame = concat(
        [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])], axis=1
    )
    train.to_csv(f"data/{file_tag}_train.csv", index=False)

    test: DataFrame = concat(
        [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])], axis=1
    )
    test.to_csv(f"data/{file_tag}_test.csv", index=False)

    return train, test

file_tag = "class_pos_covid"
target = "CovidPos"

test = split(file_tag, target)[1]
train = smotify(file_tag+"_train", target)



