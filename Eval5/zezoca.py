from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pandas import read_csv, DataFrame, Series, Index, Period, to_datetime
from matplotlib.pyplot import figure, legend, plot, savefig, show, subplot, subplots,gca, tight_layout
from sklearn.linear_model import LinearRegression
from config import FUTURE_COLOR, PAST_COLOR, PRED_FUTURE_COLOR
from dslabs_functions import FORECAST_MEASURES, plot_line_chart, HEIGHT, plot_multibar_chart, plot_multiline_chart, set_chart_labels, series_train_test_split, ts_aggregation_by
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from numpy import arange, array
from statsmodels.tsa.stattools import adfuller
from math import sqrt

# file_tag = "Traffic"
# target = "Total"
# index = "Timestamp"
# data: DataFrame = read_csv(
#     "data/Time/forecast_traffic_single.csv",
#     index_col="Timestamp",
#     sep=",",
#     decimal=".",
#     parse_dates=True,
#     infer_datetime_format=True,
# )

# series: Series = data[target]
# series.sort_index(inplace=True)
# figure(figsize=(3 * HEIGHT, HEIGHT / 2))
# plot_line_chart(
#     series.index.to_list(),
#     series.to_list(),
#     xlabel=series.index.name,
#     ylabel=target,
#     title=f"{file_tag} hourly {target}",
# )
# savefig(f"images/{file_tag}_first_granularity.png", bbox_inches='tight')
# show()

# ss_days: Series = ts_aggregation_by(series, "D")
# figure(figsize=(3 * HEIGHT, HEIGHT / 2))
# plot_line_chart(
#     ss_days.index.to_list(),
#     ss_days.to_list(),
#     xlabel="days",
#     ylabel=target,
#     title=f"{file_tag} daily mean {target}",
# )
# savefig(f"images/{file_tag}_second_granularity.png", bbox_inches='tight')
# show()

# ss_weekly: Series = ts_aggregation_by(series, "W")
# figure(figsize=(3 * HEIGHT, HEIGHT / 2))
# plot_line_chart(
#     ss_weekly.index.to_list(),
#     ss_weekly.to_list(),
#     xlabel="week",
#     ylabel=target,
#     title=f"{file_tag} weekly mean {target}",
# )
# savefig(f"images/{file_tag}_third_granularity.png", bbox_inches='tight')
# show()

# ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)

# figure(figsize=(3 * HEIGHT, HEIGHT))
# plot_line_chart(
#     ss_weeks.index.to_list(),
#     ss_weeks.to_list(),
#     xlabel="weeks",
#     ylabel=target,
#     title=f"{file_tag} weekly {target}",
# )
# show()

# ss_days: Series = ts_aggregation_by(series, gran_level="D", agg_func=sum)

# figure(figsize=(3 * HEIGHT, HEIGHT))
# plot_line_chart(
#     ss_days.index.to_list(),
#     ss_days.to_list(),
#     xlabel="days",
#     ylabel=target,
#     title=f"{file_tag} daily {target}",
# )
# show()

# fig: Figure
# axs: array
# fig, axs = subplots(2, 3, figsize=(2 * HEIGHT, HEIGHT))
# set_chart_labels(axs[0, 0], title="HOURLY")
# axs[0, 0].boxplot(series)
# set_chart_labels(axs[0, 1], title="DAILY")
# axs[0, 1].boxplot(ss_days)
# set_chart_labels(axs[0, 2], title="WEEKLY")
# axs[0, 2].boxplot(ss_weeks)

# axs[1, 0].grid(False)
# axs[1, 0].set_axis_off()
# axs[1, 0].text(0.2, 0, str(series.describe()), fontsize="small")

# axs[1, 1].grid(False)
# axs[1, 1].set_axis_off()
# axs[1, 1].text(0.2, 0, str(ss_days.describe()), fontsize="small")

# axs[1, 2].grid(False)
# axs[1, 2].set_axis_off()
# axs[1, 2].text(0.2, 0, str(ss_weeks.describe()), fontsize="small")
# savefig(f"images/{file_tag}_distribution_boxplots.png", bbox_inches='tight')
# show()


# ss_days: Series = ts_aggregation_by(series, gran_level="D", agg_func=sum)
# ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)

# grans: list[Series] = [series, ss_days, ss_weeks]
# gran_names: list[str] = ["Hourly", "Daily", "Weekly"]
# fig: Figure
# axs: array
# fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
# fig.suptitle(f"{file_tag} {target}")
# for i in range(len(grans)):
#     set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
#     axs[i].hist(grans[i].values)
# savefig(f"images/{file_tag}_distribution_histograms.png", bbox_inches='tight')
# show()

# def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
#     lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
#     for i in range(delta, max_lag + 1, delta):
#         lagged_series[f"lag {i}"] = series.shift(i)
#     return lagged_series


# figure(figsize=(3 * HEIGHT, HEIGHT))
# lags = get_lagged_series(series, 20, 10)
# plot_multiline_chart(series.index.to_list(), lags, xlabel=index, ylabel=target)
# savefig(f"images/{file_tag}_autocorrelation.png", bbox_inches='tight')
# show()

# def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
#     k: int = int(max_lag / delta)
#     fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
#     gs = GridSpec(2, k, figure=fig)

#     series_values: list = series.tolist()
#     for i in range(1, k + 1):
#         ax = fig.add_subplot(gs[0, i - 1])
#         lag = i * delta
#         ax.scatter(series.shift(lag).tolist(), series_values)
#         ax.set_xlabel(f"lag {lag}")
#         ax.set_ylabel("original")
#     ax = fig.add_subplot(gs[1, :])
#     ax.acorr(series, maxlags=max_lag)
#     ax.set_title("Autocorrelation")
#     ax.set_xlabel("Lags")
#     savefig(f"images/{file_tag}_autocorrelation.png", bbox_inches='tight')
#     show()
#     return


# autocorrelation_study(series, 10, 1)

# def plot_components(
#     series: Series,
#     title: str = "",
#     x_label: str = "time",
#     y_label: str = "",
# ) -> list[Axes]:
#     decomposition: DecomposeResult = seasonal_decompose(series, model="add", period=96)
#     components: dict = {
#         "observed": series,
#         "trend": decomposition.trend,
#         "seasonal": decomposition.seasonal,
#         "residual": decomposition.resid,
#     }
#     rows: int = len(components)
#     fig: Figure
#     axs: list[Axes]
#     fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
#     fig.suptitle(f"{title}")
#     i: int = 0
#     for key in components:
#         set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
#         axs[i].plot(components[key])
#         i += 1
#     return axs

# plot_components(
#     series,
#     title=f"{file_tag} hourly {target}",
#     x_label=series.index.name,
#     y_label=target,
# )
# savefig(f"images/{file_tag}_components_study.png", bbox_inches='tight')
# show()

# figure(figsize=(3 * HEIGHT, HEIGHT))
# plot_line_chart(
#     series.index.to_list(),
#     series.to_list(),
#     xlabel=series.index.name,
#     ylabel=target,
#     title=f"{file_tag} stationary study",
#     name="original",
# )
# n: int = len(series)
# plot(series.index, [series.mean()] * n, "r-", label="mean")
# legend()
# savefig(f"images/{file_tag}_stationary_study.png", bbox_inches='tight')
# show()

# BINS = 10
# mean_line: list[float] = []

# for i in range(BINS):
#     segment: Series = series[i * n // BINS : (i + 1) * n // BINS]
#     mean_value: list[float] = [segment.mean()] * (n // BINS)
#     mean_line += mean_value
# mean_line += [mean_line[-1]] * (n - len(mean_line))

# figure(figsize=(3 * HEIGHT, HEIGHT))
# plot_line_chart(
#     series.index.to_list(),
#     series.to_list(),
#     xlabel=series.index.name,
#     ylabel=target,
#     title=f"{file_tag} stationary study",
#     name="original",
#     show_stdev=True,
# )
# n: int = len(series)
# plot(series.index, mean_line, "r-", label="mean")
# legend()
# savefig(f"images/{file_tag}_stationary_study2.png", bbox_inches='tight')
# show()

# def eval_stationarity(series: Series) -> bool:
#     result = adfuller(series)
#     print(f"ADF Statistic: {result[0]:.3f}")
#     print(f"p-value: {result[1]:.3f}")
#     print("Critical Values:")
#     for key, value in result[4].items():
#         print(f"\t{key}: {value:.3f}")
#     return result[1] <= 0.05

# print(f"The series {('is' if eval_stationarity(series) else 'is not')} stationary")

# figure(figsize=(3 * HEIGHT, HEIGHT / 2))
# plot_line_chart(
#     series.index.to_list(),
#     series.to_list(),
#     xlabel=series.index.name,
#     ylabel=target,
#     title=f"{file_tag} hourly {target}",
# )
# figure(figsize=(3 * HEIGHT, HEIGHT / 2))
# plot_line_chart(
#     ss_agg.index.to_list(),
#     ss_agg.to_list(),
#     xlabel=ss_agg.index.name,
#     ylabel=target,
#     title=f"{file_tag} daily {target}",
# )
# plot_line_chart(
#     ss_agg2.index.to_list(),
#     ss_agg2.to_list(),
#     xlabel=ss_agg2.index.name,
#     ylabel=target,
#     title=f"{file_tag} weekly {target}",
# )

def plot_forecasting_series(
    trn: Series,
    tst: Series,
    prd_tst: Series,
    title: str = "",
    xlabel: str = "time",
    ylabel: str = "",
    ax = None
) -> list[Axes]:
    if ax is None:
        fig, ax = subplots(1, 1, figsize=(4 * HEIGHT, HEIGHT), squeeze=True)
        fig.suptitle(title)
    else:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(trn.index, trn.values, label="train", color=PAST_COLOR)
    ax.plot(tst.index, tst.values, label="test", color=FUTURE_COLOR)
    ax.plot(prd_tst.index, prd_tst.values, "--", label="test prediction", color=PRED_FUTURE_COLOR)
    ax.legend(prop={"size": 5})

    return ax

def plot_forecasting_eval(trn: Series, tst: Series, prd_trn: Series, prd_tst: Series, title: str = "", axs = None) -> list[Axes]:
    ev1: dict = {
        "RMSE": [sqrt(FORECAST_MEASURES["MSE"](trn, prd_trn)), sqrt(FORECAST_MEASURES["MSE"](tst, prd_tst))],
        "MAE": [FORECAST_MEASURES["MAE"](trn, prd_trn), FORECAST_MEASURES["MAE"](tst, prd_tst)],
    }
    ev2: dict = {
        "MAPE": [FORECAST_MEASURES["MAPE"](trn, prd_trn), FORECAST_MEASURES["MAPE"](tst, prd_tst)],
        "R2": [FORECAST_MEASURES["R2"](trn, prd_trn), FORECAST_MEASURES["R2"](tst, prd_tst)],
    }

    # print(eval1, eval2)
    if axs is None:
        fig, axs = subplots(1, 2, figsize=(1.5 * HEIGHT, 0.75 * HEIGHT), squeeze=True)
        fig.suptitle(title)
    else:
        axs[0].set_title(title)
    plot_multibar_chart(["train", "test"], ev1, ax=axs[0], title=f"Scale-dependent error ({title})", percentage=False)
    plot_multibar_chart(["train", "test"], ev2, ax=axs[1], title=f"Percentage error ({title})", percentage=True)

    return axs


filename: str = "data/time_series/forecast_covid_single.csv"
file_tag: str = "Covid"
target: str = "deaths"
timecol: str = "date"

data: DataFrame = read_csv(filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)
data.sort_index(inplace=True)
series: Series = data[target]
# series.sort_index(inplace=True)
# daily: Series = ts_aggregation_by(series, gran_level="D", agg_func="sum")
# weekly: Series = ts_aggregation_by(series, gran_level="W", agg_func="sum")

figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} hourly {target}",
)
savefig(f"images/{file_tag}_first_granularity.png", bbox_inches='tight')
show()


def train_test(data):
    train, test = series_train_test_split(data, trn_pct=0.90)
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(data)).reshape(-1, 1)
    tstY = test.to_numpy()
    model = LinearRegression()
    model.fit(trnX, trnY)
    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)
    
    return train, test, prd_trn, prd_tst

# train, test, prd_trn, prd_tst = train_test(data)
# data2 = ts_aggregation_by(data, gran_level="D", agg_func="sum")
# train2, test2, prd_trn2, prd_tst2 = train_test(data2)
# data3 = ts_aggregation_by(data, gran_level="W", agg_func="sum")
# train3, test3, prd_trn3, prd_tst3 = train_test(data3)
# fig, axs = subplots(nrows=3, ncols=1, figsize=(10, 15))
# fig.suptitle(f"{file_tag} - Aggregation - Linear Regression")
# plot_forecasting_series(
#     train,
#     test,
#     prd_tst,
#     title=f"{file_tag} - Hourly - Linear Regression",
#     xlabel=timecol,
#     ylabel=target,
#     ax = axs[0]
# )
# plot_forecasting_series(
#     train2,
#     test2,
#     prd_tst2,
#     title=f"{file_tag} - Daily - Linear Regression",
#     xlabel=timecol,
#     ylabel=target,
#     ax = axs[1]
# )
# plot_forecasting_series(
#     train3,
#     test3,
#     prd_tst3,
#     title=f"{file_tag} - Weekly - Linear Regression",
#     xlabel=timecol,
#     ylabel=target,
#     ax = axs[2]
# )
# tight_layout()
# savefig(f"images/{file_tag}_aggregation_plots.png", bbox_inches='tight')
# show()
# fig, axs = subplots(nrows=3, ncols=2, figsize=(15, 15))
# fig.suptitle(f"{file_tag} - Aggregation - Linear Regression")
# plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Hourly", axs=axs[0])
# plot_forecasting_eval(train2, test2, prd_trn2, prd_tst2, title=f"Daily", axs=axs[1])
# plot_forecasting_eval(train3, test3, prd_trn3, prd_tst3, title=f"Weekly", axs=axs[2])
# tight_layout()
# savefig(f"images/{file_tag}_aggregation_results.png", bbox_inches='tight')
# show()

# sizes: list[int] = [10, 25, 50, 75, 100]
# fig: Figure
# axs: list[Axes]
# fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
# fig.suptitle(f"{file_tag} {target} after smoothing")

# fig, axs = subplots(nrows=5, ncols=1, figsize=(10, 15))
# fig2, axs2 = subplots(nrows=5, ncols=2, figsize=(15, 15))
# fig.suptitle(f"{file_tag} - Smoothing - Linear Regression")
# fig2.suptitle(f"{file_tag} - Smoothing - Linear Regression")
# data = ts_aggregation_by(data, gran_level="D", agg_func="sum")
# for i in range(len(sizes)):
#     data2 = data.rolling(window = sizes[i]).mean()
#     # ss_smooth: Series = series.rolling(window=sizes[i]).mean()
#     # data2.dropna(inplace=True)
#     data2.fillna(0, inplace=True)
#     train, test, prd_trn, prd_tst = train_test(data2)
#     plot_forecasting_series(
#         train,
#         test,
#         prd_tst,
#         title=f"{file_tag} - size = {sizes[i]} - Linear Regression",
#         xlabel=timecol,
#         ylabel=target,
#         ax = axs[i]
#     )
#     plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"size = {sizes[i]}", axs=axs2[i])
    
# tight_layout()
# fig.savefig(f"images/{file_tag}_smoothing_plots.png", bbox_inches='tight')
# fig2.savefig(f"images/{file_tag}_smoothing_results.png", bbox_inches='tight')
# show()

# data2 = ts_aggregation_by(data, gran_level="D", agg_func="sum")
# data2 = data2.rolling(window = 25).mean()
# data2.fillna(0, inplace=True)

# fig, axs = subplots(nrows=3, ncols=1, figsize=(10, 15))
# fig2, axs2 = subplots(nrows=3, ncols=2, figsize=(15, 15))
# fig.suptitle(f"{file_tag} - Differentiation - Linear Regression")
# fig2.suptitle(f"{file_tag} - Differentiation - Linear Regression")
# train, test, prd_trn, prd_tst = train_test(data2)
# plot_forecasting_series(
#     train,
#     test,
#     prd_tst,
#     title=f"{file_tag} - Without Differentiation - Linear Regression",
#     xlabel=timecol,
#     ylabel=target,
#     ax = axs[0]
# )
# plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"Without Differentiation", axs=axs2[0])
# for i in [1, 2]:
#     data2 = data2.diff()
#     data2.fillna(0, inplace=True)
#     train, test, prd_trn, prd_tst = train_test(data2)
#     plot_forecasting_series(
#         train,
#         test,
#         prd_tst,
#         title=f"{file_tag} - {i} Differentiation - Linear Regression",
#         xlabel=timecol,
#         ylabel=target,
#         ax = axs[i]
#     )
#     plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{i} Differentiation", axs=axs2[i])
    
# tight_layout()
# fig.savefig(f"images/{file_tag}_differentiation_plots.png", bbox_inches='tight')
# fig2.savefig(f"images/{file_tag}_differentiation_results.png", bbox_inches='tight')
# show()