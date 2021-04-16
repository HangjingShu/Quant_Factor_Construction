import pandas as pd
import numpy as np
import pandas as pd
import datetime as dt

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_分析师预期 13
class EquFactorAf(Equ):
    def __init__(self, bic, sb, ac):
        super().__init__()
        self.bic = bic
        self.sb = sb
        self.ac = ac
### bic: 将财报中的balance sheet, income statement, cash flow三张表的数据合并而成
### sb: stock_basic, 股票进阶数据,如PE等
### ac: 分析师预测数据
### OHLC: 股票后复权数据，如每日收盘价、开盘价、最高点、最低点、换手率等
### OHLC_qfq：股票前复权数据
### 未来可优化的点：使用SQL直接从数据库抓取stock_basic中的trade_date数据，并直接建表，可以避免每只股票重复读取sb表
### 尽量用矩阵代替for的遍历函数，但目前为止还未想出好的办法
### 有些reindex以及copy的指令冗余，未来可以进一步删除不必要的部分

    def Con_Profit_Wind(self):
        """
        因子描述：
        使用万得算法的分析师一致预期归母净利润
        算法：算数平均法
        样本：全部卖方机构研报（去除异常值，主要是数据过大过小错误）,去除重复研报数据
        有效期：最新披露原则（机构最新的研报替代之前的研报），180天失效（滚动180天）（需对原表日期进行填充）
        """
        DATE = self.df_sb.copy().reset_index("trade_date")[["trade_date"]].copy()
        DATE["trade_date"] = pd.to_datetime(DATE["trade_date"])
        DATE["re"]=0
        N = self.df_ac.copy().drop_duplicates(subset=["research_report_title", "forecast_object"], keep="first").reset_index()
        N = N[N["forecast_object"] == "PROFIT"].reset_index()
        for i in range(len(DATE["trade_date"])):
            t = DATE["trade_date"][i] - dt.timedelta(days = 180)
            a = N[(N["create_date"] < DATE["trade_date"][i]) & (N["create_date"] >= t)]
            a = a.sort_values("create_date").reset_index()
            Nprofit = a.groupby(["time_year", "organ_name"])[["forecast1", "forecast2"]].last().reset_index(drop = False)
            for j in range(len(Nprofit["time_year"])):
                if Nprofit.loc[j, "time_year"] == DATE.loc[i, "trade_date"].year:
                    pass
                elif (Nprofit.loc[j, "time_year"]+1) == DATE.loc[i, "trade_date"].year:
                    Nprofit.loc[j, "forecast1"] = Nprofit.loc[j, "forecast2"]
                else:
                    Nprofit.loc[j, "forecast1"] = np.nan
            DATE.loc[i, "re"]=Nprofit["forecast1"].mean()
        re = DATE.set_index(self.df_sb.index)["re"]
        return re

    def Con_EPS_Wind(self):
        """
        因子描述：
        使用万得算法的分析师一致预期每股收益
        算法：算数平均法
        样本：全部卖方机构研报（去除异常值，主要是数据过大过小错误）
        有效期：最新披露原则（机构最新的研报替代之前的研报），180天失效（滚动180天）（需对原表日期进行填充）
        """
        DATE = self.df_sb.copy().reset_index("trade_date")[["trade_date"]].copy()
        DATE["trade_date"] = pd.to_datetime(DATE["trade_date"])
        DATE["re"]=0
        N = self.df_ac.copy().drop_duplicates(subset=["research_report_title", "forecast_object"], keep="first").reset_index()
        N = N[N["forecast_object"] == "EPS"].reset_index()
        for i in range(len(DATE["trade_date"])):
            t = DATE["trade_date"][i] - dt.timedelta(days = 180)
            a = N[(N["create_date"] < DATE["trade_date"][i]) & (N["create_date"] >= t)]
            a = a.sort_values("create_date").reset_index()
            EPS = a.groupby(["time_year", "organ_name"])[["forecast1", "forecast2"]].last().reset_index(drop = False)
            for j in range(len(EPS["time_year"])):
                if EPS.loc[j, "time_year"] == DATE.loc[i, "trade_date"].year:
                    pass
                elif (EPS.loc[j, "time_year"]+1) == DATE.loc[i, "trade_date"].year:
                    EPS.loc[j, "forecast1"] = EPS.loc[j, "forecast2"]
                else:
                    EPS.loc[j, "forecast1"] = np.nan
            DATE.loc[i, "re"] = EPS["forecast1"].mean()
        re = DATE.set_index(self.df_sb.index)["re"]
        return re

    def Con_Income_Wind(self):
        """
        因子描述：
        使用万得算法的分析师一致预期营业收入
        算法：算数平均法
        有效期：最新披露原则（机构最新的研报替代之前的研报），180天失效（滚动180天）
        """
        DATE = self.df_sb.copy().reset_index("trade_date")[["trade_date"]].copy()
        DATE["trade_date"] = pd.to_datetime(DATE["trade_date"])
        DATE["re"]=0
        N = self.df_ac.copy().drop_duplicates(subset=["research_report_title", "forecast_object"], keep="first").reset_index()
        N = N[N["forecast_object"] == "INCOME"].reset_index()
        for i in range(len(DATE["trade_date"])):
            t = DATE["trade_date"][i] - dt.timedelta(days = 180)
            a = N[(N["create_date"] < DATE["trade_date"][i]) & (N["create_date"] >= t)]
            a = a.sort_values("create_date").reset_index()
            income = a.groupby(["time_year", "organ_name"])[["forecast1", "forecast2"]].last().reset_index(drop = False)
            for j in range(len(income["time_year"])):
                if income.loc[j, "time_year"] == DATE.loc[i, "trade_date"].year:
                    pass
                elif (income.loc[j, "time_year"]+1) == DATE.loc[i, "trade_date"].year:
                    income.loc[j, "forecast1"] = income.loc[j, "forecast2"]
                else:
                    income.loc[j, "forecast1"] = np.nan
            DATE.loc[i, "re"] = income["forecast1"].mean()
        re = DATE.set_index(self.df_sb.index)["re"]
        return re

    def Con_Profit_weighted(self):
        """
        因子描述：
        分析师一致预期归母净利润
        算法：加权平均法，权重为每个机构在T-1年的预测偏度，如果机构在T-1年有多条预测记录，则对预测偏度取均值。
        预测偏度=(abs(预测-实际))/(abs(实际))
        具体赋权过程为把所有机构按其预测偏度均值从小到大排序分成5档，分别赋予权重5，4，3，2，1，作为该机构的综合预测权重（所有股票一起）
        有效期：滚动180天，最后把各个月份(取30天为1个月)的预期值由近至远赋予半衰权重32，16，8，4，2，1 即可求得该股票的一致预期数据。
        """
        DATE = self.df_sb.copy().reset_index("trade_date")[["trade_date"]].copy()
        DATE["trade_date"] = pd.to_datetime(DATE["trade_date"])
        DATE["re"] = 0
        N = self.ac.processed_data.copy()
        N = N[N["forecast_object"] == "PROFIT"].copy()
        N["deviation"] = abs((N["forecast1"] - N["true_Value"])) / abs(N["true_Value"])
        deviation_mean = N.groupby(["time_year", "author_name"])["deviation"].mean().reset_index(
            drop=False).sort_values(["author_name", "time_year"]).reset_index()
        deviation_mean["lag"] = deviation_mean.groupby("author_name")["deviation"].shift(1)
        deviation_mean = deviation_mean.dropna(axis=0)
        deviation_mean["signal"] = deviation_mean.groupby("time_year")["lag"].transform("count")
        deviation_mean = deviation_mean[deviation_mean["signal"] >= 5].copy()
        deviation_mean["weight"] = deviation_mean.groupby("time_year")["lag"].apply(lambda x: pd.qcut(x.rank(method="first"),5,labels=range(5,0,-1)))
        t = deviation_mean[["author_name", "time_year", "weight"]].copy()
        a = self.df_ac
        a = a[a["forecast_object"] == "PROFIT"].copy().reset_index()
        fin = pd.merge(a, t, on=["time_year", "author_name"], how="left").reset_index()
        fin["mon"] = 0
        for i in range(len(DATE["trade_date"])):
            t = DATE["trade_date"][i] - dt.timedelta(days=180)
            samp = fin[(fin["create_date"] < DATE["trade_date"][i]) & (fin["create_date"] >= t)].reset_index()
            for e in range(1,7):
                tim = t + dt.timedelta(days=30) * e
                tim_l = t + dt.timedelta(days=30) * (e - 1)
                bo = (samp["create_date"] < tim) & (samp["create_date"] >= tim_l)
                samp.loc[bo, "mon"] = e
        #### 比如2021-02-03这天往前推半年可能有去年的数据，这个可能有预测2020年的，如果time_year和trade date的year相等（因为我们只要trade date那一年的预测数据）
        ####  pass，比他小1去看forecast2，有就取过来
            for j in range(len(samp["time_year"])):
                if samp.loc[j, "time_year"] == DATE.loc[i, "trade_date"].year:
                    pass
                elif (samp.loc[j, "time_year"] + 1) == DATE.loc[i, "trade_date"].year:
                    samp.loc[j, "forecast1"] = samp.loc[j, "forecast2"]
                else:
                    samp.loc[j, "forecast1"] = np.nan
            samp=samp.dropna(axis=0)
            samp["weight"]=samp["weight"].astype("int")
            samp["weighted"] = samp["weight"] / samp["weight"].sum()
            samp["weighted_ana"] = samp["forecast1"] * samp["weighted"]
        #### 半衰权重理解为直接加权做，就是每过一个月信息权重减半
            m_con = samp.groupby("mon")["weighted_ana"].sum().reset_index()
            m_con["half"] = 2** (m_con["mon"] - 1)
            m_con["half_weight"] = m_con["half"] / m_con["half"].sum()
            m_con["s"] = m_con["half_weight"] * m_con["weighted_ana"]
            DATE.loc[i, "re"] = m_con["s"].sum()
        re = DATE.set_index(self.df_sb.index)["re"]
        return re

    def Con_EPS_weighted(self):
        """
        因子描述：
        分析师一致预期每股收益
        算法：加权平均法，权重为每个机构在T-1年的预测偏度，如果机构在T-1年有多条预测记录，则对预测偏度取均值。
        预测偏度=(abs(预测-实际))/(abs(实际))
        具体赋权过程为把所有机构按其预测偏度均值从小到大排序分成5档，分别赋予权重5，4，3，2，1，作为该机构的综合预测权重（所有股票一起）
        有效期：滚动180天，最后把各个月份(取30天为1个月)的预期值由近至远赋予半衰权重32，16，8，4，2，1 即可求得该股票的一致预期数据。
        """
        DATE = self.df_sb.copy().reset_index("trade_date")[["trade_date"]].copy()
        DATE["trade_date"] = pd.to_datetime(DATE["trade_date"])
        DATE["re"] = 0
        N = self.ac.processed_data.copy()
        N = N[N["forecast_object"] == "EPS"].copy()
        N["deviation"] = abs((N["forecast1"] - N["true_Value"])) / abs(N["true_Value"])
        deviation_mean = N.groupby(["time_year", "author_name"])["deviation"].mean().reset_index(
            drop=False).sort_values(["author_name", "time_year"]).reset_index()
        deviation_mean["lag"] = deviation_mean.groupby("author_name")["deviation"].shift(1)
        deviation_mean = deviation_mean.dropna(axis=0)
        deviation_mean["signal"] = deviation_mean.groupby("time_year")["lag"].transform("count")
        deviation_mean = deviation_mean[deviation_mean["signal"] >= 5].copy()
        deviation_mean["weight"] = deviation_mean.groupby("time_year")["lag"].apply(lambda x: pd.qcut(x.rank(method="first"),5,labels=range(5,0,-1)))
        t = deviation_mean[["author_name", "time_year", "weight"]].copy()
        a = self.df_ac
        a = a[a["forecast_object"] == "EPS"].copy().reset_index()
        fin = pd.merge(a, t, on=["time_year", "author_name"], how="left").reset_index()
        fin["mon"] = 0
        for i in range(len(DATE["trade_date"])):
            t = DATE["trade_date"][i] - dt.timedelta(days=180)
            samp = fin[(fin["create_date"] < DATE["trade_date"][i]) & (fin["create_date"] >= t)].reset_index()
            for e in range(1, 7):
                tim = t + dt.timedelta(days=30) * e
                tim_l = t + dt.timedelta(days=30) * (e - 1)
                bo = (samp["create_date"] < tim) & (samp["create_date"] >= tim_l)
                samp.loc[bo, "mon"] = e
            for j in range(len(samp["time_year"])):
                if samp.loc[j, "time_year"] == DATE.loc[i, "trade_date"].year:
                    pass
                elif (samp.loc[j, "time_year"] + 1) == DATE.loc[i, "trade_date"].year:
                    samp.loc[j, "forecast1"] = samp.loc[j, "forecast2"]
                else:
                    samp.loc[j, "forecast1"] = np.nan
            samp = samp.dropna(axis=0)
            samp["weight"] = samp["weight"].astype("int")
            samp["weighted"] = samp["weight"] / samp["weight"].sum()
            samp["weighted_ana"] = samp["forecast1"] * samp["weighted"]
            m_con = samp.groupby("mon")["weighted_ana"].sum().reset_index()
            m_con["half"] = 2 ** (m_con["mon"] - 1)
            m_con["half_weight"] = m_con["half"] / m_con["half"].sum()
            m_con["s"] = m_con["half_weight"] * m_con["weighted_ana"]
            DATE.loc[i, "re"] = m_con["s"].sum()
        re = DATE.set_index(self.df_sb.index)["re"]
        return re

    def Con_Income_weighted(self):
        """
        因子描述：
        分析师一致预期营业收入
        算法：加权平均法，权重为每个机构在T-1年的预测偏度，如果机构在T-1年有多条预测记录，则对预测偏度取均值。
        预测偏度=(abs(预测-实际))/(abs(实际))
        具体赋权过程为把所有机构按其预测偏度均值从小到大排序分成5档，分别赋予权重5，4，3，2，1，作为该机构的综合预测权重（所有股票一起）
        有效期：滚动180天，最后把各个月份(取30天为1个月)的预期值由近至远赋予半衰权重32，16，8，4，2，1 即可求得该股票的一致预期数据。
        """
        DATE = self.df_sb.copy().reset_index("trade_date")[["trade_date"]].copy()
        DATE["trade_date"] = pd.to_datetime(DATE["trade_date"])
        DATE["re"]=0
        N = self.ac.processed_data.copy()
        N = N[N["forecast_object"] == "INCOME"].copy()
        N["deviation"] = abs((N["forecast1"] - N["true_Value"])) / abs(N["true_Value"])
        deviation_mean = N.groupby(["time_year", "author_name"])["deviation"].mean().reset_index(
            drop=False).sort_values(["author_name", "time_year"]).reset_index()
        deviation_mean["lag"] = deviation_mean.groupby("author_name")["deviation"].shift(1)
        deviation_mean = deviation_mean.dropna(axis=0)
        deviation_mean["signal"] = deviation_mean.groupby("time_year")["lag"].transform("count")
        deviation_mean = deviation_mean[deviation_mean["signal"] >= 5].copy()
        deviation_mean["weight"] = deviation_mean.groupby("time_year")["lag"].apply(lambda x: pd.qcut(x.rank(method="first"),5,labels=range(5,0,-1)))
        t = deviation_mean[["author_name", "time_year", "weight"]].copy()
        a = self.df_ac
        a = a[a["forecast_object"] == "INCOME"].copy().reset_index()
        fin = pd.merge(a, t, on=["time_year", "author_name"], how="left").reset_index()
        fin["mon"] = 0
        for i in range(len(DATE["trade_date"])):
            t = DATE["trade_date"][i] - dt.timedelta(days=180)
            samp = fin[(fin["create_date"] < DATE["trade_date"][i]) & (fin["create_date"] >= t)].reset_index()
            for e in range(1, 7):
                tim = t + dt.timedelta(days=30) * e
                tim_l = t + dt.timedelta(days=30) * (e - 1)
                bo = (samp["create_date"] < tim) & (samp["create_date"] >= tim_l)
                samp.loc[bo, "mon"] = e
            for j in range(len(samp["time_year"])):
                if samp.loc[j, "time_year"] == DATE.loc[i, "trade_date"].year:
                    pass
                elif (samp.loc[j, "time_year"] + 1) == DATE.loc[i, "trade_date"].year:
                    samp.loc[j, "forecast1"] = samp.loc[j, "forecast2"]
                else:
                    samp.loc[j, "forecast1"] = np.nan
            samp = samp.dropna(axis=0)
            samp["weight"] = samp["weight"].astype("int")
            samp["weighted"] = samp["weight"] / samp["weight"].sum()
            samp["weighted_ana"] = samp["forecast1"] * samp["weighted"]
            m_con = samp.groupby("mon")["weighted_ana"].sum().reset_index()
            m_con["half"] = 2 ** (m_con["mon"] - 1)
            m_con["half_weight"] = m_con["half"] / m_con["half"].sum()
            m_con["s"] = m_con["half_weight"] * m_con["weighted_ana"]
            DATE.loc[i, "re"] = m_con["s"].sum()
        re = DATE.set_index(self.df_sb.index)["re"]
        return re

    # def REC(self):
    #     """
    #     因子描述：
    #     分析师推荐评级（rating score by analyst）。
    #     计算方法：
    #     用卖方“Go-Goal 评级”，买入、增持、中性、减持、卖出赋予分值分别为：1、0.75、0.5、0.25、0，形成“go-goal 评级强度”值数列。
    #     数值越高，买入信号越强。
    #     """
    #     DATE = self.df_sb.copy().reset_index("trade_date")[["trade_date"]].copy()
    #     DATE["trade_date"] = pd.to_datetime(DATE["trade_date"])
    #     re = self.df_sb.copy().reset_index("trade_date")["trade_date"].copy()
    #
    #
    # def DAREC(self):
    #     """
    #     因子描述：
    #     分析师推荐评级变化（Changes of recommended rating score by analyst），相比于60 个交易日前。
    #     计算方法：
    #     DAREC t = REC t - REC (t-60)
    #     """
    #
    # def GREC(self):
    #     """
    #     因子描述：
    #     分析师推荐评级变化趋势（Change tendency of recommended rating score by analyst），过去60个交易日内的
    #     DAREC 符号加和。
    #     计算方法：
    #
    #     """
    #
    # def FY12P(self):
    #     """
    #     因子描述：
    #     分析师盈利预测（Forecast earnings by analyst to market values）。
    #     计算方法：
    #
    #     其中 Earing 是指一致预期归属母公司净利润（万），MarketValue是总市值
    #     注：Earnings为当年的预期值，如20180121日的因子中，预期值为2018年的利润
    #     注：分母<0时，因子值为空
    #     """
    #
    # def DAREV(self):
    #     """
    #     因子描述：
    #     分析师盈利预测变化（Changes of forecast earnings by analyst），相比于60个交易日前。
    #     计算方法：
    #
    #     其中Earnings表示一致预期归属母公司净利润（万）, 无论是t日还是t-60日都是对t日所属年份的预期值
    #     """
    #
    # def GREV(self):
    #     """
    #     因子描述：
    #     分析师盈利预测变化趋势（Change tendency of forecast earnings by analyst），过去60 个交易日内的DAREV符号加和。
    #     计算方法：
    #
    #     其中Earnings表示一致预期归属母公司净利润（万）,无论是t日还是t-60日都是对t日所属年份的预期值
    #     """
    #
    # def SFY12P(self):
    #     """
    #     因子描述：
    #     分析师营收预测（Forecast sales by analyst to market values）。
    #     计算方法：
    #     SFY12P=(一致预期预测收入（万, Latest）×10000)/总市值(Latest)
    #     若流通股本数值缺失使用总股本数值代替。
    #     注：Sales为当年的预期值，如20180121日的因子中，预期值为2018年的收入
    #     注：分母<0时，因子值为空
    #     """
    #
    # def DASREV(self):
    #     """
    #     因子描述：
    #     分析师盈收预测变化（Changes of forecast sales by analyst (to 60 days ago)），相比于60个交易日前。
    #     计算方法：
    #
    #     其中sales表示一致预期预测收入（万）,无论是t日还是t-60日都是对t日所属年份的预期值
    #     """
    #
    # def GSREV(self):
    #     """
    #     因子描述：
    #     分析师盈收预测变化趋势（Change tendency of forecast sales by analyst, Sum of 60 days' DASREV），过去60个交易日内的DASREV 符号加和。
    #     计算方法：
    #
    #     其中sales表示一致预期预测收入（万）
    #     """
    #
    # def FEARNG(self):
    #     """
    #     因子描述：
    #     未来预期盈利增长（Forecasted growth rate of earnings ）。
    #     计算方法：
    #
    #     Earnings是指一致预期归属母公司净利润（万）
    #     """
    #
    # def FSALESG(self):
    #     """
    #     因子描述：
    #     未来预期盈收增长（Forecasted growth rate of sales）。
    #     计算方法：
    #
    #     Sales是指一致预期预测收入（万）
    #     """
    #
    # def EPIBS(self):
    #     """
    #     因子描述：
    #     投资回报率预测（Forecast earnings by analyst to market values）。
    #     计算方法：
    #     EPIBS=一致预期归属母公司净利润（万, Latest）×10000/总市值(Latest)
    #     注: 预期值为因子日当年的预期值，如20180121日的因子中，预期值年份为2018
    #     注：分母<0时，因子值为空
    #     """
    #
    # def EgibsLong(self):
    #     """
    #     因子描述：
    #     长期盈利增长预测（Long-term Predicted Earnings Growth）。
    #     计算方法：
    #
    #     Earnings是指一致预期归属母公司净利润（万）
    #     """
