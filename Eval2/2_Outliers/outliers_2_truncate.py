from pandas import read_csv, DataFrame, concat
from dslabs_functions import get_variable_types, encode_cyclic_variables, dummify
from pandas import read_csv, DataFrame, Series
from dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)

file_tag = "class_pos_covid_frequent_fill"
files = [ f"data/{file_tag}.csv"]
data: DataFrame = concat([read_csv(file, index_col=None) for file in files], axis=0)

print(f"Original data: {data.shape}")

n_std: int = NR_STDEV
numeric_vars: list[str] = get_variable_types(data)["numeric"]
summary5: DataFrame = data[numeric_vars].describe()

if [] != numeric_vars:
    df: DataFrame = data.copy(deep=True)
    for var in numeric_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        df[var] = df[var].apply(
            lambda x: top if x > top else bottom if x < bottom else x
        )
    df.to_csv(f"data/{file_tag}_truncate_outliers.csv", index=True)
    print("Data after truncating outliers:", df.shape)
    print(df.describe())
else:
    print("There are no numeric variables")


from dslabs_functions import evaluate_approach, plot_multibar_chart
from matplotlib.pyplot import savefig, show, figure

target = "CovidPos"
train = df.sample(frac=0.9, random_state=1)
test = df.drop(train.index)
figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"images/{file_tag}_eval_truncate_outliers.png")
show()
