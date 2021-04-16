import pandas as pd
import numpy as np

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_估值与市值 28
class EquFactorVs(Equ):
    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb

    def PE(self):
        """
        因子描述： 市盈率（Price-earnings ratio），使用TTM算法。
            计算方法： 市盈率=总市值/归属于母公司所有者的净利润。
            归属于母公司所有者的净利润按TTM 计算。
        """
        PE1 = self.df_sb["pe_ttm"]
        return PE1

    def PS(self):
        """
        因子描述： 市销率（Price-to-sales ratio）。
        计算方法： PS = 总市值/营业收入
        """
        PS1 = self.df_sb["ps"]
        return PS1

    def PCF(self):
        """
        因子描述： 市现率（Price-to-cash-flow ratio）。
        计算方法： 市现率 = 总市值 / 经营活动产生的现金流量净额
            其中营活动产生的现金流量净额取TTM。
        """
        tmv = self.df_sb["total_mv"]
        nca = self.df_bic['n_cashflow_act']
        nca = self.expand_index(self.drop_duplicated_ann_date(nca), self.df_sb)
        PCF1 = tmv / nca
        return PCF1

    def PB(self):
        """
        因子描述： 市净率（Price-to-book ratio）。
        计算方法： 市净率=总市值/(归属于母公司所有者权益合计-其他权益合计)。
        “其他权益工具”包括“优先股”，“永续债”等不归属于普通股股东的所有者权益。
        """
        PB1 = self.df_sb["pb"]
        return PB1

    def ASSI(self):
        """
        总资产 = 资产总计
        因子描述： 对数总资产（Natural logarithm of total assets）。
        计算方法： 对数总资产=总资产的对数log10
        注：因子值<0时，因子值为空
        """
        t_assets = self.df_bic['total_assets'].fillna(0)
        ASSI_tmp = t_assets.apply(np.log10)
        ASSI1 = self.expand_index(self.drop_duplicated_ann_date(ASSI_tmp), self.df_sb)
        return ASSI1

    def LCAP(self):
        """
        因子描述： 对数市值（Natural logarithm of total market values）。
        计算方法： 对数市值=总市值的对数 log10
        注：因子值<0时，因子值为空
        """
        LCAP1 = self.df_sb['total_mv'].fillna(0)
        LCAP1 = LCAP1.apply(np.log10)
        LCAP1[LCAP1 < 0] = np.nan
        return LCAP1

    def LFLO(self):
        """
        因子描述： 对数流通市值（Natural logarithm of negotiable market value ）。
        计算方法： 对数流通市值=流通市值的对数 流动市值缺失时用总市值代替
        """
        LFLO1 = self.df_sb['circ_mv'].fillna(0)
        LFLO1 = LFLO1.apply(np.log10)
        LFLO1[LFLO1 < 0] = np.nan
        return LFLO1

    def TA2EV(self):
        """
        因子描述： 资产总计与企业价值之比（Assets to enterprise value）。
        计算方法： TA2EV = 总资产/企业价值 = 总资产/（长期借款+短期借款+总市值-现金及现金等价物）。
        """
        lt_borr = self.df_bic["lt_borr"].fillna(0)
        st_borr = self.df_bic["st_borr"].fillna(0)
        equ_end_period = self.df_bic["c_cash_equ_end_period"].fillna(0)
        t_assets = self.df_bic['total_assets'].fillna(0)
        LT = self.expand_index(self.drop_duplicated_ann_date(lt_borr), self.df_sb)
        ST = self.expand_index(self.drop_duplicated_ann_date(st_borr), self.df_sb)
        EquEnd = self.expand_index(self.drop_duplicated_ann_date(equ_end_period), self.df_sb)
        TA = self.expand_index(self.drop_duplicated_ann_date(t_assets), self.df_sb)
        tmv = self.df_sb["total_mv"]
        TA2EV1 = TA/(LT + ST + tmv - EquEnd)
        return TA2EV1

    def PEG3Y(self):
        """
        因子描述： 市盈率/归属于母公司所有者净利润3年复合增长率。
        计算方法：
        特别说明：当PE为负值和净利润复合增长率均为负值时，PEG3Y的因子值也是负数。
        """
        PE1 = self.df_sb["pe"]
        NP_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        NP_3_TTM = NP_TTM.shift(periods=12, fill_value=0)
        net_profit_grow_rate = np.sign(NP_TTM) * ((NP_TTM / NP_3_TTM).abs().pow(1 / 3, fill_value=0)) - 1
        net_profit_grow_rate_3Y = self.drop_duplicated_ann_date(net_profit_grow_rate)
        NetProfitGrowRate3Y = self.expand_index(net_profit_grow_rate_3Y, self.df_sb)
        PEG3Y1 = PE1 / NetProfitGrowRate3Y
        return PEG3Y1

    def PEG5Y(self):
        """
        因子描述： 市盈率/归属于母公司所有者净利润5年复合增长率。
        计算方法：
        特别说明：当PE为负值和净利润复合增长率均为负值时，PEG5Y的因子值也是负数。
        """
        PE1 = self.df_sb["pe"]
        NP_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        NP_5_TTM = NP_TTM.shift(periods=20, fill_value=0)
        net_profit_grow_rate = np.sign(NP_TTM) * ((NP_TTM / NP_5_TTM).abs().pow(1 / 5, fill_value=0)) - 1
        net_profit_grow_rate_5Y = self.drop_duplicated_ann_date(net_profit_grow_rate)
        NetProfitGrowRate5Y = self.expand_index(net_profit_grow_rate_5Y, self.df_sb)
        PEG5Y1 = PE1 / NetProfitGrowRate5Y
        return PEG5Y1

    def PEIndu(self):
        """
        因子描述： （PE – PE的行业均值）/PE的行业标准差。
        计算方法：
        注：由于PE会出现负值，和极大极小值的情况，所以先对原始数据去极值（winsorize,5%），
            然后去掉PE为负的值，最后再按照上面的公式计算因子值
        """

    def PBIndu(self):
        """
        因子描述： （PB – PB的行业均值）/PB的行业标准差。
        计算方法：
        注：由于PB会出现负值，和极大极小值的情况，所以先对原始数据去极值（winsorize,5%），
            然后去掉PB为负的值，最 后再按照上面的公式计算因子值

        """

    def PSIndu(self):
        """
        因子描述： （PS – PS的行业均值）/PS的行业标准差。
        计算方法：
        注：由于PS会出现负值，和极大极小值的情况，所以先对原始数据去极值（winsorize,5%），
            然后去掉PS为负的值，最 后再按照上面的公式计算因子值；
        """

    def PCFIndu(self):
        """
        因子描述： （PCF – PCF的行业均值）/PCF的行业标准差。
        计算方法：
        注：由于PCF会出现负值，和极大极小值的情况，所以先对原始数据去极值（winsorize,5%），
            然后去掉PCF为负的值， 最后再按照上面的公式计算因子值
        """

    def PEHist20(self):
        """
        因子描述：个股PE/过去一个月个股PE的均值。
        计算方法：
        其中PE是指动态市盈率，取自PE因子的值
        """
        PE1 = self.df_sb["pe"]
        PE_20mean = self.df_sb["pe"].rolling(20).mean()
        PEHist_20 = PE1 / PE_20mean
        return PEHist_20

    def PEHist60(self):
        """
        因子描述：个股PE/过去三个月个股PE的均值。
        计算方法：
        其中PE是指动态市盈率，取自PE因子的值
        """
        PE1 = self.df_sb["pe"]
        PE_60mean = self.df_sb["pe"].rolling(60).mean()
        PEHist_60 = PE1 / PE_60mean
        return PEHist_60

    def PEHist120(self):
        """
        因子描述：个股PE/过去六个月个股PE的均值。
        计算方法：
        其中PE是指动态市盈率，取自PE因子的值
        """
        PE1 = self.df_sb["pe"]
        PE_120mean = self.df_sb["pe"].rolling(120).mean()
        PEHist_120 = PE1 / PE_120mean
        return PEHist_120

    def PEHist250(self):
        """
        因子描述：个股PE/过去一年个股PE的均值。
        计算方法：
        其中PE是指动态市盈率，取自PE因子的值
        """
        PE1 = self.df_sb["pe"]
        PE_250mean = self.df_sb["pe"].rolling(250).mean()
        PEHist_250 = PE1 / PE_250mean
        return PEHist_250

    def StaticPE(self):
        """
        因子描述： 静态PE。
        计算方法： 等于总市值/归属于母公司所有者的净利润，净利润只取年报数据
        """


    def ForwardPE(self):
        """
        因子描述： 动态PE。
        计算方法： 等于总市值/归属于母公司所有者的净利润，净利润根据每期财报动态扩展到年报数据，
        假设三季报净利润为A，那么分 母为4/3*A。
        """

    def TotalAssets(self):
        """
        因子描述： 总资产。
        计算方法： 总资产的最新值。
        """
        t_assets = self.df_bic['total_assets'].fillna(0)
        t_assets = self.expand_index(self.drop_duplicated_ann_date(t_assets), self.df_sb)
        return t_assets

    def MktValue(self):
        """
        因子描述： 总市值。
        计算方法： 总市值的最新值。
        """
        MktValue1 = self.df_sb['total_mv'].fillna(0)
        return MktValue1

    def NegMktValue(self):
        """
        因子描述： 流通市值。
        计算方法： 流通市值当时=可交易的流通股股数×收盘价
        注：因子值<0时，因子值为空
        """
        NegMktValue1 = self.df_sb['circ_mv'].fillna(0)
        return NegMktValue1

    def TEAP(self):
        """
        因子描述： 归属于母公司所有者权益。
        计算方法： 最新的归属于母公司所有者权益
        """
        TEAP1 = self.df_bic["total_hldr_eqy_exc_min_int"]
        TEAP1 = self.expand_index(self.drop_duplicated_ann_date(TEAP1), self.df_sb)
        return TEAP1

    def NIAP(self):
        """
        因子描述： 归属于母公司所有者的净利润。 ​
        计算方法： 归属于母公司所有者的净利润的最新值。
        """
        NIAP1 = self.df_bic["n_income_attr_p"]
        NIAP1 = self.expand_index(self.drop_duplicated_ann_date(NIAP1), self.df_sb)
        return NIAP1

    def CETOP(self):
        """
        因子描述： 现金收益滚动收益与市值比（Cash earnings-to-price ratio）。
        计算方法： Computed by dividing the trailing 12-month cash earnings divided by current price.
        """
        NCA = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        NCA = self.expand_index(self.drop_duplicated_ann_date(NCA), self.df_sb)
        MktValue1 = self.df_sb['total_mv'].fillna(0)
        CETOP1 = NCA / MktValue1
        return CETOP1

    def SGRO(self):
        """
        因子描述：5年营业总收入增长率（Five-year sales growth）。
        计算方法：5年总营收增长率= 5年总营收关于时间（年）进行线性回归的回归系数/5年总营业收入均值的绝对值
        其中OR取年报数据，t是时间，是5年总营收关于时间（年）进行线性回归的回归系数，mean(OR) 是5年总营收均值。
        进行线性回归时，当数据库中记录数大于五年时取最近五年的数据，记录数不足五年时取全部数据，最后结果仅保留记录数不小于三年的数据。
        """
    #     需要选取年报数据

    def NLSIZE(self):
        """
        因子描述： 对数市值立方。
        计算方法： 先对市值取对数，然后再计算对数市值的立方
        注：因子值<0时，因子值为空
        """
        LCAP1 = self.df_sb['total_mv'].fillna(0)
        LCAP1 = LCAP1.apply(np.log10)
        NLSIZE1 = LCAP1 ** 3
        NLSIZE1[NLSIZE1 < 0] = np.nan
        return NLSIZE1

