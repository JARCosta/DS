from pandas import read_csv, DataFrame, concat
from dslabs_functions import get_variable_types, encode_cyclic_variables, dummify

files = [ "data/class_pos_covid.csv", "data/class_pos_covid2.csv"]
file_tag = "class_pos_covid"
data: DataFrame = concat([read_csv(file, index_col=None) for file in files], axis=0)

vars: dict[str, list] = get_variable_types(data)

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}

encoding: dict[str, dict[str, int]] = {
    "Sex": {'Female': 0, 'Male': 1},
    "GeneralHealth": {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Poor': 3, 'Very good': 4},
    "LastCheckupTime": {'5 or more years ago': 5, 'Within past 2 years (1 year but less than 2 years ago)': 1, 'Within past 5 years (2 years but less than 5 years ago)': 2, 'Within past year (anytime less than 12 months ago)': 0},
    "PhysicalActivities": yes_no,
    "RemovedTeeth": {'1 to 5': 1, '6 or more, but not all': 6, 'All': 20, 'None of them': 0},
    "HadHeartAttack": yes_no,
    "HadAngina": yes_no,
    "HadStroke": yes_no,
    "HadAsthma": yes_no,
    "HadSkinCancer": yes_no,
    "HadCOPD": yes_no,
    "HadDepressiveDisorder": yes_no,
    "HadKidneyDisease": yes_no,
    "HadArthritis": yes_no,
    "HadDiabetes": {'No': 0, 'No, pre-diabetes or borderline diabetes': 1, 'Yes': 3, 'Yes, but only during pregnancy (female)': 2},
    "DeafOrHardOfHearing": yes_no,
    "BlindOrVisionDifficulty": yes_no,
    "DifficultyConcentrating": yes_no,
    "DifficultyWalking": yes_no,
    "DifficultyDressingBathing": yes_no,
    "DifficultyErrands": yes_no,
    "SmokerStatus": {'Current smoker - now smokes every day': 3, 'Current smoker - now smokes some days': 2, 'Former smoker': 1, 'Never smoked': 0},
    "ECigaretteUsage": {'Never used e-cigarettes in my entire life': 0, 'Not at all (right now)': 1, 'Use them every day': 2, 'Use them some days': 3},
    "ChestScan": yes_no,
    "AgeCategory": {'Age 18 to 24': 18, 'Age 25 to 29': 25, 'Age 30 to 34': 30, 'Age 35 to 39': 35, 'Age 40 to 44': 40, 'Age 45 to 49': 45, 'Age 50 to 54': 50, 'Age 55 to 59': 55, 'Age 60 to 64': 60, 'Age 65 to 69': 65, 'Age 70 to 74': 70, 'Age 75 to 79': 75, 'Age 80 or older': 80},
    "AlcoholDrinkers": yes_no,
    "HIVTesting": yes_no,
    "FluVaxLast12": yes_no,
    "PneumoVaxEver": yes_no,
    "TetanusLast10Tdap": {'No, did not receive any tetanus shot in the past 10 years': 0, 'Yes, received tetanus shot but not sure what type': 1, 'Yes, received tetanus shot, but not Tdap': 2, 'Yes, received Tdap': 3},
    "HighRiskLastYear": yes_no,
    "CovidPos": yes_no,

}
df: DataFrame = data.replace(encoding, inplace=False)
df.head()
data = df


# IDs
# Binary variables linear
# State dummify
# GeneralHealth Linear
# LastCheckup Linear? dummify
# SmokerStatus taxo
# ECigarette taxo
# race dummify
# age linear
# tetanus taxo

from numpy import ndarray
from pandas import DataFrame, read_csv, concat
from sklearn.preprocessing import OneHotEncoder

from dslabs_functions import dummify

# vars: list[str] = ["State", "RaceEthnicityCategory"]
# # vars: list[str] = ["RaceEthnicityCategory"]
# df: DataFrame = dummify(data, vars)
# data = df
# df.head()

df.drop(columns=["State", "RaceEthnicityCategory", "HeightInMeters", "Sex", "WeightInKilograms", ], inplace=True)

print("dropping and filling missing values ")

from dslabs_functions import mvi_by_dropping

df: DataFrame = mvi_by_dropping(data, min_pct_per_variable=0.9, min_pct_per_record=0.9)
data = df


from dslabs_functions import mvi_by_filling

df: DataFrame = mvi_by_filling(data, strategy="frequent")
data = df
df.to_csv(f"data/{file_tag}_frequent_fill.csv", index=False)

print("creating plot")

from dslabs_functions import evaluate_approach, plot_multibar_chart
from matplotlib.pyplot import savefig, show, figure

target = "CovidPos"
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.1, random_state=42)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"images/{file_tag}_eval_frequent_fill_delete_5_no_dummies.png")
show()

# print("analyzing correlations")

# from dslabs_functions import get_variable_types
# from matplotlib.pyplot import figure, subplots, savefig, show
# from seaborn import heatmap

# variables_types: dict[str, list] = get_variable_types(data)
# all_vars = []
# for i in variables_types:
#     all_vars.extend(variables_types[i])
# # numeric: list[str] = variables_types["numeric"]
# numeric: list[str] = all_vars
# corr_mtx: DataFrame = data[numeric].corr().abs()

# figure()
# heatmap(
#     abs(corr_mtx),
#     xticklabels=numeric,
#     yticklabels=numeric,
#     annot=False,
#     cmap="Blues",
#     vmin=0,
#     vmax=1,
# )
# savefig(f"images/{file_tag}_correlation_analysis.png")
# show()




