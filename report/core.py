import numpy as np
import pandas as pd


class Performance(object):
    def __init__(self, values):
        self.__values = values
        self.__max_drawdown = np.nan
        self.__drawdown_details = np.nan
        self.__sharpe_ratio = np.nan
        self.__win_rate = np.nan
        self.__cagr = np.nan
        self.__calmar_ratio = np.nan
        self.__drawdown_series = self.__to_drawdown_series()
        self.__return_series = self.__values / self.__values.shift(1) - 1.0

    def __to_drawdown_series(self):
        drawdown = self.__values.copy()
        drawdown.fillna(method='pad', inplace=True)

        drawdown[np.isnan(drawdown)] = -np.Inf

        roll_max = np.maximum.accumulate(drawdown)
        drawdown = drawdown / roll_max - 1.0

        return drawdown

    @property
    def max_drawdown(self):
        return self.__max_drawdown

    @max_drawdown.getter
    def max_drawdown(self):
        dd_details = self.drawdown_details

        return dd_details.ix[dd_details.drawdown.idxmin()]

    @property
    def drawdown_details(self):
        return self.__drawdown_details

    @drawdown_details.getter
    def drawdown_details(self):
        drawdown = self.__drawdown_series
        is_zero = drawdown == 0

        start = ~is_zero & is_zero.shift(1)
        start = list(start[start].index)

        end = is_zero & (~is_zero).shift(1)
        end = list(end[end].index)

        if len(start) is 0:
            return None

        if len(end) is 0:
            end.append(drawdown.index[-1])

        if start[0] > end[0]:
            start.insert(0, drawdown.index[0])

        if start[-1] > end[-1]:
            end.append(drawdown.index[-1])

        result = pd.DataFrame(columns=('start', 'lowest', 'end', 'down', 'up', 'drawdown'),
                              index=range(0, len(start)))

        for i in range(0, len(start)):
            dd = drawdown[start[i]:end[i]].min()
            idx = drawdown[start[i]:end[i]].idxmin()
            result.ix[i] = (start[i], idx, end[i], (idx - start[i]).days, (end[i] - idx).days, dd)

        return result

    @property
    def win_rate(self):
        return self.__win_rate

    @win_rate.getter
    def win_rate(self):
        win = self.__values.diff()
        rate = np.sum(win > 0) / (len(win > 0) + len(win < 0))
        return rate

    @property
    def sharpe_ratio(self):
        return self.__sharpe_ratio

    @sharpe_ratio.getter
    def sharpe_ratio(self):
        ratio = self.__return_series.mean() / self.__return_series.std() * np.sqrt(252)
        return ratio

    @property
    def cagr(self):
        return self.__cagr

    @cagr.getter
    def cagr(self):
        start = self.__values.first_valid_index()
        end = self.__values.last_valid_index()
        total_second = (end - start).total_seconds() / 31557600.0

        cagr_rate = (self.__values.ix[end] / self.__values.ix[start]) ** (1.0 / total_second) - 1.0

        return cagr_rate

    @property
    def calmar_ratio(self):
        return self.__calmar_ratio

    @calmar_ratio.getter
    def calmar_ratio(self):
        ratio = self.cagr / self.max_drawdown.drawdown

        return ratio

    def win_rate_by(self, symbol):
        """
        win rate in day, week, month frequency
        :param symbol: D for Day; W for week; M for month
        :return: win rate in symbol frequency
        """
        result = self.__values.resample(symbol, how='last')
        result = result.diff()

        wining_rate = np.sum(result > 0.0) / np.sum(result != 0.0)
        return wining_rate

    def rolling_return_by(self, n):
        """
        rolling return information in n days
        :param n: rolling window length
        :return:
        """
        daily_result = self.__values.resample('D', how='last')
        daily_result.dropna(axis=0, inplace=True)

        roll_return = pd.rolling_apply(daily_result, n, lambda x: x[-1] / x[0] - 1.0)
        return {
            'max': roll_return.max(),
            'min': roll_return.min(),
            'mean': roll_return.mean()
        }
