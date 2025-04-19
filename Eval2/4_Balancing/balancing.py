from pandas import read_csv, concat, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig
from dslabs_functions import plot_bar_chart

file_tag = "class_pos_covid_frequent_fill"
files = [ f"data/{file_tag}.csv"]
original: DataFrame = concat([read_csv(file, index_col=None) for file in files], axis=0)
target = "CovidPos"

target_count: Series = original[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()

print("Minority class=", positive_class, ":", target_count[positive_class])
print("Majority class=", negative_class, ":", target_count[negative_class])
print(
    "Proportion:",
    round(target_count[positive_class] / target_count[negative_class], 2),
    ": 1",
)
values: dict[str, list] = {
    "Original": [target_count[positive_class], target_count[negative_class]]
}

# figure()
# plot_bar_chart(
#     target_count.index.to_list(), target_count.to_list(), title="Class balance"
# )
# savefig(f"images/{file_tag}_class_balance.png")


df_positives: Series = original[original[target] == positive_class]
df_negatives: Series = original[original[target] == negative_class]



# df_neg_sample: DataFrame = DataFrame(df_negatives.sample(len(df_positives)))
# df_under: DataFrame = concat([df_positives, df_neg_sample], axis=0)
# df_under.to_csv(f"data/{file_tag}_under.csv", index=False)

# print("Minority class=", positive_class, ":", len(df_positives))
# print("Majority class=", negative_class, ":", len(df_neg_sample))
# print("Proportion:", round(len(df_positives) / len(df_neg_sample), 2), ": 1")


# from dslabs_functions import evaluate_approach, plot_multibar_chart

# train = df_under.sample(frac=0.9, random_state=1)
# test = df_under.drop(train.index)

# figure()
# eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
# plot_multibar_chart(
#     ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
# )
# savefig(f"images/{file_tag}_eval_under.png")
# print("Under-sampling evaluation")


# df_pos_sample: DataFrame = DataFrame(
#     df_positives.sample(len(df_negatives), replace=True)
# )
# df_over: DataFrame = concat([df_pos_sample, df_negatives], axis=0)
# df_over.to_csv(f"data/{file_tag}_over.csv", index=False)

# print("Minority class=", positive_class, ":", len(df_pos_sample))
# print("Majority class=", negative_class, ":", len(df_negatives))
# print("Proportion:", round(len(df_pos_sample) / len(df_negatives), 2), ": 1")


# from dslabs_functions import evaluate_approach, plot_multibar_chart

# train = df_over.sample(frac=0.9, random_state=1)
# test = df_over.drop(train.index)

# figure()
# eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
# plot_multibar_chart(
#     ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
# )
# savefig(f"images/{file_tag}_eval_over.png")
# print("Over-sampling evaluation")



from numpy import ndarray
from pandas import Series
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = original.pop(target).values
X: ndarray = original.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(original.columns) + [target]
df_smote.to_csv(f"data/{file_tag}_smote.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print(
    "Proportion:",
    round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
    ": 1",
)
print(df_smote.shape)


from dslabs_functions import evaluate_approach, plot_multibar_chart

train = df_smote.sample(frac=0.9, random_state=1)
test = df_smote.drop(train.index)

figure()
eval: dict[str, list] = evaluate_approach(train, test, target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"images/{file_tag}_eval_smote.png")
print("SMOTE evaluation")

