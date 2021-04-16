import pandas as pd

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_每股指标 24 - 17
class EquFactorPsi(Equ):
    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb
        self.index = self.sb.fetch()

    def EPS(self):
        """
        因子描述：
        基本每股收益（Earnings per share）。
        计算方法：
        基本每股收益 (income) = 归属于母公司的净利润（动态年化） / 总股本
        分子归属于母公司的净利润根据每期财报动态扩展到年报数据，假设当期是三季报的归母净利润E， 则分子为4/3E。
        """
        basic_eps = self.drop_duplicated_ann_date(self.df_bic["basic_eps"])
        EPS = self.expand_index(basic_eps, self.df_sb)

        return EPS

    def DilutedEPS(self):
        """
        因子描述：
        稀释每股收益（Diluted earnings per share, income）。
        计算方法：
        取财报稀释每股收益的结果(income)。
        """
        diluted_eps = self.drop_duplicated_ann_date(self.df_bic["diluted_eps"])
        DilutedEPS = self.expand_index(diluted_eps, self.df_sb)

        return DilutedEPS

    def CashDividendCover(self):
        # TODO
        """
        因子描述：
        现金股利保障倍数。
        计算方法：
        现金股利保障倍数 = 最近一次除权除息日对应的经营活动产生的现金流量净额TTM / （最近一次每股派现税前 ×分红前总股本）
        若某股票过去两年无派现，则该股票无该值
        **现金股利保障倍数=每股营业现金流量 / 每股现金股利=经营活动现金净流量 / 现金股利
        """

    def DividendCover(self):
        # TODO
        """
        因子描述：
        股利保障倍数。
        计算方法：
        股利保障倍数 = 最近一次除权除息日对应的归属于母公司所有者的净利润TTM / （最近一次每股派现税前 ×分红前总股本）
        若某股票过去两年无派现，则该股票无该值
        **股利保障倍数=（净利润总额-优先股股利总额）/普通股股利总额
        """

    def DividendPaidRatio(self):
        # TODO
        """
        因子描述：
        股利支付率。
        计算方法：
        股利支付率 = （最近一次每股派现税前 ×分红前总股本）/归属于母公司所有者的净利润TTM
        注1：当净利润TTM为负或者过去两年公司无现金分红时，该值不计算。
        注2：实际情况中也存在现金股利总额大于净利润，所以会出现股利支付率大于1的情况。
        **股利支付率=每股股利÷每股净收益×100%=股利总额÷净利润总额
        """

    def RetainedEarningRatio(self):
        # TODO
        """
        因子描述：
        留存盈余比率。
        计算方法：
        留存盈余比率 = 1 - （最近一次每股派现税前 × 分红前总股本） / 归属于母公司所有者的净利润TTM
        注：计算时，直接取DividendPaidRatio的因子值， RetainedEarningRatio = 1-DividendPaidRatio
        """

    def CashEquivalentPS(self):
        """
        因子描述：
        每股现金及现金等价物余额。
        计算方法：
        每股现金及现金等价物余额 = 期末现金及现金等价物余额(Latest, cashflow) / 总股本(Latest, stock_basic)
        其中分子取Latest，不需要进行TTM处理；分母需要根据当时日期来获取最近变更日的总股本。
        财务科目如有空值，则用前值填充
        """
        equ_end_period = self.drop_duplicated_ann_date(self.df_bic["c_cash_equ_end_period"].fillna(method="pad"))
        ECF = self.expand_index(equ_end_period, self.df_sb)
        TS = self.df_sb["total_share"]
        CashEquivalentPS = ECF / TS
        return CashEquivalentPS

    def DividendPS(self):
        # TODO
        """
        因子描述：
        每股股利。
        计算方法：
        每股股利（税前）= 母公司应付股利 / 母公司总股本
        数据直接取财报中的每股股利（税前），以公告日期为准；若某公司在过去连续两年均未分红，该公司不计算此指标。
        过去连续两年表示分红发布日期pub_date要大于当前日期对应的两年前
        """

    def EPSTTM(self):
        # TODO
        """
        因子描述：
        每股收益TTM。
        计算方法：
        每股收益TTM = 归属于母公司所有者的净利润(TTM, income) / 总股本(stock_basic,Latest)
        """
        attr_p_TTM = self.df_bic["n_income_attr_p"].rolling(4, min_periods=0).mean()
        attr_p = self.drop_duplicated_ann_date(attr_p_TTM)
        NPAP = self.expand_index(attr_p, self.df_sb)
        TS = self.df_sb["total_share"]
        EPSTTM = NPAP / TS
        return EPSTTM

    def NetAssetPS(self):
        # TODO
        """
        因子描述：
        每股净资产。
        计算方法：
        每股净资产 = 归属于母公司所有者权益合计(Latest) / 总股本(stock_basic,Latest)
        TSEP为归母净利润-其它权益工具
        """
        total_hldr_eqy_exc_min_int = self.drop_duplicated_ann_date(self.df_bic["total_hldr_eqy_exc_min_int"])
        TSEP = self.expand_index(total_hldr_eqy_exc_min_int, self.df_sb)
        TS = self.df_sb["total_share"]
        NetAssetPS = TSEP / TS
        return NetAssetPS

    def TORPS(self):
        """
        因子描述：
        每股营业总收入。
        计算方法：
        每股营业总收入 = 营业总收入(TTM, income) / 总股本(Latest, stock_basic)
        """
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        total_revenue = self.drop_duplicated_ann_date(total_revenue_TTM)
        TOR = self.expand_index(total_revenue, self.df_sb)
        TS = self.df_sb["total_share"]
        TORPS = TOR / TS
        return TORPS

    def TORPSLatest(self):
        """
        因子描述：
        每股营业总收入（最新）。
        计算方法：
        每股营业总收入（最新） = 营业总收入(Latest, income) / 总股本(Latest, stock_basic)
        分子营业总收入根据每期财报动态扩展到年报数据，假设当期是三季报的营业总收入R， 则分子为4/3R
        """
        total_revenue = self.drop_duplicated_ann_date(self.df_bic["total_revenue"])
        TOR = self.expand_index(total_revenue, self.df_sb)
        TS = self.df_sb["total_share"]
        TORPSLatest = TOR / TS
        return TORPSLatest

    def OperatingRevenuePS(self):
        """
        因子描述：
        每股营业收入。
        计算方法：
        每股营业收入 =  营业收入(TTM, income) / 总股本(Latest, stock_basic)
        """
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        revenue = self.drop_duplicated_ann_date(revenue_TTM)
        OR = self.expand_index(revenue, self.df_sb)
        TS = self.df_sb["total_share"]
        OperatingRevenuePS = OR / TS
        return OperatingRevenuePS

    def OperatingRevenuePSLatest(self):
        """
        因子描述：
        每股营业收入（最新）。
        计算方法：
        每股营业收入（最新） = 营业收入(Latest) / 总股本(Latest, stock_basic)
        """
        revenue = self.drop_duplicated_ann_date(self.df_bic["revenue"])
        OR = self.expand_index(revenue, self.df_sb)
        TS = self.df_sb["total_share"]
        OperatingRevenuePSLatest = OR / TS
        return OperatingRevenuePSLatest

    def OperatingProfitPS(self):
        """
        因子描述：
        每股营业利润。
        计算方法：
        每股营业利润 = 营业利润(TTM, income) / 总股本(Latest, stock_basic)
        """
        operate_profit_TTM = self.df_bic["operate_profit"].rolling(4, min_periods=0).mean()
        operate_profit = self.drop_duplicated_ann_date(operate_profit_TTM)
        OP = self.expand_index(operate_profit, self.df_sb)
        TS = self.df_sb["total_share"]
        OperatingProfitPS = OP / TS
        return OperatingProfitPS

    def OperatingProfitPSLatest(self):
        """
        因子描述：
        每股营业利润（最新）。
        计算方法：
        每股营业利润（最新） = 营业利润(Latest, income) / 总股本(Latest, stock_basic)
        分子营业利润根据每期财报动态扩展到年报数据，假设当期是三季报的营业利润E， 则分子为4/3E
        """
        operate_profit = self.drop_duplicated_ann_date(self.df_bic["operate_profit"])
        OP = self.expand_index(operate_profit, self.df_sb)
        TS = self.df_sb["total_share"]
        OperatingProfitPSLatest = OP / TS
        return OperatingProfitPSLatest

    def CapitalSurplusFundPS(self):
        """
        因子描述：
        每股资本公积金。
        计算方法：
        每股资本公积金 = 资本公积(Latest, balancesheet) / 总股本(Latest, stock_basic)
        财务科目的空值用前值填充
        """
        cap_rese = self.drop_duplicated_ann_date(self.df_bic["cap_rese"].fillna(method="pad"))
        CAPITAL_RESER = self.expand_index(cap_rese, self.df_sb)
        TS = self.df_sb["total_share"]
        CapitalSurplusFundPS = CAPITAL_RESER / TS
        return CapitalSurplusFundPS

    def SurplusReserveFundPS(self):
        """
        因子描述：
        每股盈余公积金。
        计算方法：
        每股盈余公积金 = 盈余公积(Latest, balancesheet) / 总股本(Latest, stock_basic)
        财务科目的空值用前值填充
        """
        surplus_rese = self.drop_duplicated_ann_date(self.df_bic["surplus_rese"].fillna(method="pad"))
        SURPLUS_RESER = self.expand_index(surplus_rese, self.df_sb)
        TS = self.df_sb["total_share"]
        SurplusReserveFundPS = SURPLUS_RESER / TS
        return SurplusReserveFundPS

    def UndividedProfitPS(self):
        """
        因子描述：
        每股未分配利润。
        计算方法：
        每股未分配利润 = 未分配利润(Latest, balancesheet) / 总股本(Latest, stock_basic)
        """
        undistr_porfit = self.drop_duplicated_ann_date(self.df_bic["undistr_porfit"])
        RETAINED_EARNINGS = self.expand_index(undistr_porfit, self.df_sb)
        TS = self.df_sb["total_share"]
        UndividedProfitPS = RETAINED_EARNINGS / TS
        return UndividedProfitPS

    def RetainedEarningsPS(self):
        """
        因子描述：
        每股留存收益。
        计算方法：
        每股留存收益 = ( 盈余公积(Latest) + 未分配利润(Latest, balancesheet) ) / 总股本(Latest, stock_basic)
        财务科目的空值用前值填充
        """
        surplus_rese = self.df_bic["surplus_rese"].fillna(method="pad")
        undistr_porfit = self.df_bic["undistr_porfit"].fillna(method="pad")
        SUM = surplus_rese.add(undistr_porfit, fill_value=0)
        add_surplus_undistr = self.drop_duplicated_ann_date(SUM)
        ADD_SURPLUS_RETAINED = self.expand_index(add_surplus_undistr, self.df_sb)
        TS = self.df_sb["total_share"]
        # ADD_SURPLUS_RETAINED = SURPLUS_RESER + RETAINED_EARNINGS
        RetainedEarningsPS = ADD_SURPLUS_RETAINED / TS
        return RetainedEarningsPS

    def OperCashFlowPS(self):
        """
        因子描述：
        每股经营活动产生的现金流量净额。
        计算方法：
        每股经营活动产生的现金流量净额 = 经营活动产生的现金流量净额(TTM, cashflow) / 总股本(Latest, stock_basic)
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        act = self.drop_duplicated_ann_date(act_TTM)
        NOCF = self.expand_index(act, self.df_sb)
        TS = self.df_sb["total_share"]
        OperCashFlowPS = NOCF / TS
        return OperCashFlowPS

    def CashFlowPS(self):
        """
        因子描述：
        每股现金流量净额。
        计算方法：
        每股现金流量净额 = 现金及现金等价物净增加额(TTM, cashflow) / 总股本(Latest, stock_basic)
        """
        cash_equ_TTM = self.df_bic["n_incr_cash_cash_equ"].rolling(4, min_periods=0).mean()
        cash_equ = self.drop_duplicated_ann_date(cash_equ_TTM)
        NCC = self.expand_index(cash_equ, self.df_sb)
        TS = self.df_sb["total_share"]
        CashFlowPS = NCC / TS
        return CashFlowPS

    def EnterpriseFCFPS(self):
        # TODO
        """
        因子描述：
        每股企业自由现金流量。
        计算方法：
        从公告中获取的每股企业自由现金流量。
        """

    def ShareholderFCFPS(self):
        # TODO
        """
        因子描述：
        每股股东自由现金流量。
        计算方法：
        从公告中获取的每股股东自由现金流量。
        """
