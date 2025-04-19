from pandas import concat
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split
from dslabs_functions import plot_multibar_chart




from pandas import read_csv
from numpy import array, ndarray
from pandas import read_csv, DataFrame


def read_train_test_from_files(
    train_fn: str, test_fn: str, target: str = "class"
) -> tuple[ndarray, ndarray, array, array, list, list]:
    train: DataFrame = read_csv(train_fn, index_col=None)
    labels: list = list(train[target].unique())
    labels.sort()
    trnY: array = train.pop(target).to_list()
    trnX: ndarray = train.values

    test: DataFrame = read_csv(test_fn, index_col=None)
    tstY: array = test.pop(target).to_list()
    tstX: ndarray = test.values
    return trnX, tstX, trnY, tstY, labels, train.columns.to_list()




file_tag = "class_pos_covid"
train_filename = f"data/{file_tag}_smote_train.csv"
test_filename = f"data/{file_tag}_frequent_fill_test.csv"
target = "CovidPos"
eval_metric = "accuracy"



def smote(file_tag: str, target: str = "class"):


    from pandas import read_csv, concat, DataFrame, Series
    from matplotlib.pyplot import figure, show
    from dslabs_functions import plot_bar_chart


    original: DataFrame = read_csv(f"data/{file_tag}_frequent_fill_train.csv", sep=",", decimal=".")

    target_count: Series = original[target].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

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
    df_smote.to_csv(f"data/{file_tag}_smote_train.csv", index=False)

    smote_target_count: Series = Series(smote_y).value_counts()
    print("Minority class=", positive_class, ":", smote_target_count[positive_class])
    print("Majority class=", negative_class, ":", smote_target_count[negative_class])
    print(
        "Proportion:",
        round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),
        ": 1",
    )
    print(df_smote.shape)

trnX: ndarray
tstX: ndarray
trnY: array
tstY: array
labels: list
vars: list
trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)
print(f"Train#={trnX.shape} Test#={tstX.shape}")
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")





from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(trnX, trnY)
pred_trnY: array = clf.predict(trnX)
print(f"Score over Train: {clf.score(trnX, trnY):.3f}")
print(f"Score over Test: {clf.score(tstX, tstY):.3f}")




from sklearn.metrics import accuracy_score, recall_score, precision_score

pred_tstY: array = clf.predict(tstX)

acc: float = accuracy_score(tstY, pred_tstY)
recall: float = recall_score(tstY, pred_tstY)
prec: float = precision_score(tstY, pred_tstY)
print(f"accuracy={acc:.3f} recall={recall:.3f} precision={prec:.3f}")







from pandas import unique
from sklearn.metrics import confusion_matrix

labels: list = list(unique(tstY))
labels.sort()

prdY: array = clf.predict(tstX)
cnf_mtx_tst: ndarray = confusion_matrix(tstY, prdY, labels=labels)
print(cnf_mtx_tst)






from itertools import product
from numpy import ndarray, set_printoptions, arange
from matplotlib.pyplot import gca, cm
from matplotlib.axes import Axes


def plot_confusion_matrix(cnf_matrix: ndarray, classes_names: ndarray, ax: Axes = None) -> Axes:  # type: ignore
    if ax is None:
        ax = gca()
    title = "Confusion matrix"
    set_printoptions(precision=2)
    tick_marks: ndarray = arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cnf_matrix, interpolation="nearest", cmap=cm.Blues)

    for i, j in product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        ax.text(
            j, i, format(cnf_matrix[i, j], "d"), color="y", horizontalalignment="center"
        )
    return ax


figure()
plot_confusion_matrix(cnf_mtx_tst, labels)
show()








from sklearn.metrics import RocCurveDisplay
from config import ACTIVE_COLORS


def plot_roc_chart(tstY: ndarray, predictions: dict, ax: Axes = None, target: str = "class") -> Axes:  # type: ignore
    if ax is None:
        ax = gca()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("FP rate")
    ax.set_ylabel("TP rate")
    ax.set_title("ROC chart for %s" % target)

    ax.plot(
        [0, 1],
        [0, 1],
        color="navy",
        label="random",
        linewidth=1,
        linestyle="--",
        marker="",
    )
    models = list(predictions.keys())
    for i in range(len(models)):
        RocCurveDisplay.from_predictions(
            y_true=tstY,
            y_pred=predictions[models[i]],
            name=models[i],
            ax=ax,
            color=ACTIVE_COLORS[i],
            linewidth=1,
        )
    ax.legend(loc="lower right", fontsize="xx-small")
    return ax


figure()
plot_roc_chart(tstY, {"GaussianNB": prdY}, target=target)
show()











from typing import Callable
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, figure
from sklearn.metrics import roc_auc_score, f1_score
from dslabs_functions import plot_multibar_chart, HEIGHT

CLASS_EVAL_METRICS: dict[str, Callable] = {
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "auc": roc_auc_score,
    "f1": f1_score,
}


def plot_evaluation_results(
    model, trn_y, prd_trn, tst_y, prd_tst, labels: ndarray
) -> ndarray:
    evaluation: dict = {}
    for key in CLASS_EVAL_METRICS:
        evaluation[key] = [
            CLASS_EVAL_METRICS[key](trn_y, prd_trn),
            CLASS_EVAL_METRICS[key](tst_y, prd_tst),
        ]

    params_st: str = "" if () == model["params"] else str(model["params"])
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    fig.suptitle(f'Best {model["metric"]} for {model["name"]} {params_st}')
    plot_multibar_chart(["Train", "Test"], evaluation, ax=axs[0], percentage=True)

    cnf_mtx_tst: ndarray = confusion_matrix(tst_y, prd_tst, labels=labels)
    plot_confusion_matrix(cnf_mtx_tst, labels, ax=axs[1])
    return axs


model_description: dict = {"name": "GaussianNB", "metric": eval_metric, "params": ()}

prd_trn: array = clf.predict(trnX)
prd_tst: array = clf.predict(tstX)
figure()
plot_evaluation_results(model_description, trnY, prd_trn, tstY, prd_tst, labels)
savefig(
    f'images/{file_tag}_{model_description["name"]}_best_{model_description["metric"]}_eval.png'
)
show()