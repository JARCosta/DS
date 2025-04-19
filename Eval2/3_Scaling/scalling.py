from pandas import read_csv, DataFrame, Series, concat
from sklearn.preprocessing import StandardScaler



file_tag = "class_pos_covid_frequent_fill"
files = [ f"data/{file_tag}.csv"]
data: DataFrame = concat([read_csv(file, index_col=None) for file in files], axis=0)

target = "CovidPos"
vars: list[str] = data.columns.to_list()
print(vars)
target_data: Series = data.pop(target)

print("Z-score normalization")

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore[target] = target_data
df_zscore.columns = vars
# df_zscore.to_csv(f"data/{file_tag}_scaled_zscore.csv", index="id")

print("Z-score normalization evaluation")

from dslabs_functions import evaluate_approach, plot_multibar_chart
from matplotlib.pyplot import figure, show, savefig

train = df_zscore.sample(frac=0.9, random_state=1)
test = df_zscore.drop(train.index)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["KNN", ], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"images/{file_tag}_eval_scaled_zscore.png")


'''

print("MinMax normalization")

from sklearn.preprocessing import MinMaxScaler

transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
# df_minmax.to_csv(f"data/{file_tag}_scaled_minmax.csv", index="id")

print("MinMax normalization evaluation")

from dslabs_functions import evaluate_approach, plot_multibar_chart
from matplotlib.pyplot import figure, show, savefig

train = df_minmax.sample(frac=0.9, random_state=1)
test = df_minmax.drop(train.index)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["KNN", ], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"images/{file_tag}_eval_scaled_minmax.png")



print("creating plots")

from matplotlib.pyplot import subplots, show

fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 1].set_title("Original data")
data.boxplot(ax=axs[0, 0])
axs[0, 0].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
show()

'''