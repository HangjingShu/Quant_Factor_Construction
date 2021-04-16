import statsmodels.api as sm
import pandas as pd
import numpy as np

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_盈利能力和收益质量 37 - 14
class EquFactorPq(Equ):
    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb

    def NetProfitRatio(self):
        """
        因子描述：
        销售净利率（Net profit ratio）。
        计算方法：
        销售净利率 = 净利润(TTM, income) / 营业收入(TTM, income)
        """
        net_profit_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        net_profit_ratio_TTM = net_profit_TTM / revenue_TTM
        net_profit_ratio = self.drop_duplicated_ann_date(net_profit_ratio_TTM)
        NetProfitRatio = self.expand_index(net_profit_ratio, self.df_sb)
        return NetProfitRatio

    def OperatingProfitRatio(self):
        """
        因子描述：
        营业利润率（Operating profit ratio）。
        计算方法：
        营业利润率 = 营业利润(TTM, income) / 营业收入(TTM, income)
        """
        operate_profit_TTM = self.df_bic["operate_profit"].rolling(4, min_periods=0).mean()
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        operating_profit_ratio_TTM = operate_profit_TTM / revenue_TTM
        operating_profit_ratio = self.drop_duplicated_ann_date(operating_profit_ratio_TTM)
        OperatingProfitRatio = self.expand_index(operating_profit_ratio, self.df_sb)
        return OperatingProfitRatio

    def NPToTOR(self):
        """
        因子描述：
        净利润与营业总收入之比（Net profit to total revenues）。
        计算方法：
        净利润与营业总收入之比 = 净利润(TTM, income) / 营业总收入(TTM, income)
        """
        net_profit_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        np_to_tor_TTM = net_profit_TTM / total_revenue_TTM
        np_to_tor = self.drop_duplicated_ann_date(np_to_tor_TTM)
        NPToTOR = self.expand_index(np_to_tor, self.df_sb)
        return NPToTOR

    def OperatingProfitToTOR(self):
        """
        因子描述：
        营业利润与营业总收入之比（Operating profit to total revenues）。
        计算方法：
        营业利润与营业总收入之比 = 营业利润(TTM, income) / 营业总收入(TTM, income)
        """
        operate_profit_TTM = self.df_bic["operate_profit"].rolling(4, min_periods=0).mean()
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        operating_profit_to_tor_TTM = operate_profit_TTM / total_revenue_TTM
        operating_profit_to_tor = self.drop_duplicated_ann_date(operating_profit_to_tor_TTM)
        OperatingProfitToTOR = self.expand_index(operating_profit_to_tor, self.df_sb)
        return OperatingProfitToTOR

    def GrossIncomeRatio(self):
        """
        因子描述：
        销售毛利率（Gross income ratio）。
        计算方法：
        销售毛利率=( 营业收入(TTM, income) - 营业成本(TTM, income) ) / 营业收入(TTM, income)
        oper_cost
        """
        oper_cost_TTM = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean().fillna(0)
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean().fillna(0)
        gross_income_ratio_TTM = (revenue_TTM - oper_cost_TTM) / revenue_TTM
        gross_income_ratio = self.drop_duplicated_ann_date(gross_income_ratio_TTM)
        GrossIncomeRatio = self.expand_index(gross_income_ratio, self.df_sb)
        return GrossIncomeRatio

    def SalesCostRatio(self):
        """
        因子描述：
        销售成本率（Sales cost ratio）。
        计算方法：
        销售成本率 = 营业成本(TTM, income) / 营业收入(TTM, income)
        """
        oper_cost_TTM = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        sales_cost_ratio_TTM = oper_cost_TTM / revenue_TTM
        sales_cost_ratio = self.drop_duplicated_ann_date(sales_cost_ratio_TTM)
        SalesCostRatio = self.expand_index(sales_cost_ratio, self.df_sb)
        return SalesCostRatio

    def TaxRatio(self):
        """
        因子描述：
        销售税金率（Tax ratio）。
        计算方法：
        销售税金率 = 营业税金及附加(TTM, income)  / 营业收入(TTM, income)
        """
        biz_tax_surchg_TTM = self.df_bic["biz_tax_surchg"].rolling(4, min_periods=0).mean()
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        tax_ratio_TTM = biz_tax_surchg_TTM / revenue_TTM
        tax_ratio = self.drop_duplicated_ann_date(tax_ratio_TTM)
        TaxRatio = self.expand_index(tax_ratio, self.df_sb)
        return TaxRatio

    def EBITToTOR(self):
        """
        因子描述：
        息税前利润与营业总收入之比（Earnings before interest and tax to total revenues）。
        计算方法:
        息税前利润与营业总收入之比 = ( 利润总额(TTM, income) + 利息支出(TTM, income) - 利息收入(TTM, income) ) / 营业总收入(TTM, income)
        如果没有利息支出，用财务费用代替
        int_exp fin_exp int_income total_revenue
        """
        total_profit_TTM = self.df_bic["total_profit"].rolling(4, min_periods=0).mean().fillna(0)
        int_exp_TTM = self.df_bic["int_exp"].rolling(4, min_periods=0).mean().fillna(0)
        int_income_TTM = self.df_bic["int_income"].rolling(4, min_periods=0).mean().fillna(0)
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        EBITtoTOR = (total_profit_TTM + int_exp_TTM - int_income_TTM) / total_revenue_TTM
        t = self.drop_duplicated_ann_date(EBITtoTOR)
        EBITToTOR = self.expand_index(t, self.df_sb)
        return EBITToTOR

    def FinancialExpenseRate(self):
        """
        因子描述：
        财务费用与营业总收入之比（Financial expense rate）。
        计算方法：
        财务费用与营业总收入之比 = 财务费用(TTM, income) / 营业总收入(TTM, income)
        """
        fin_exp_TTM = self.df_bic["fin_exp"].rolling(4, min_periods=0).mean()
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        financial_expense_rate_TTM = fin_exp_TTM / total_revenue_TTM
        financial_expense_rate = self.drop_duplicated_ann_date(financial_expense_rate_TTM)
        FinancialExpenseRate = self.expand_index(financial_expense_rate, self.df_sb)
        return FinancialExpenseRate

    def OperatingExpenseRate(self):
        """
        因子描述：
        营业费用与营业总收入之比（Operating expense rate）。
        计算方法：
        营业费用与营业总收入之比 = 销售费用(TTM, income) / 营业总收入(TTM, income)
        """
        sell_exp_TTM = self.df_bic["sell_exp"].rolling(4, min_periods=0).mean()
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        operating_expense_rate_TTM = sell_exp_TTM / total_revenue_TTM
        operating_expense_rate = self.drop_duplicated_ann_date(operating_expense_rate_TTM)
        OperatingExpenseRate = self.expand_index(operating_expense_rate, self.df_sb)
        return OperatingExpenseRate

    def AdminiExpenseRate(self):
        """
        因子描述：
        管理费用与营业总收入之比（Administrative expense rate）。
        计算方法：
        管理费用与营业总收入之比 = 管理费用(TTM, income) / 营业总收入(TTM, income)
        """
        admin_exp_TTM = self.df_bic["admin_exp"].rolling(4, min_periods=0).mean()
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        admini_expense_rate_TTM = admin_exp_TTM / total_revenue_TTM
        admini_expense_rate = self.drop_duplicated_ann_date(admini_expense_rate_TTM)
        AdminiExpenseRate = self.expand_index(admini_expense_rate, self.df_sb)
        return AdminiExpenseRate

    def TotalProfitCostRatio(self):
        """
        因子描述：
        成本费用利润率（Total profit cost ratio）。
        计算方法：
        成本费用利润率 = 利润总额(TTM, income) / ( 营业成本(TTM, income)+财务费用(TTM, income)+销售费用(TTM, income)+管理费用(TTM, income) )
        """
        total_profit_TTM = self.df_bic["total_profit"].rolling(4, min_periods=0).mean()
        oper_cost_TTM = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean().fillna(0)
        fin_exp_TTM = self.df_bic["fin_exp"].rolling(4, min_periods=0).mean().fillna(0)
        sell_exp_TTM = self.df_bic["sell_exp"].rolling(4, min_periods=0).mean().fillna(0)
        admin_exp_TTM = self.df_bic["admin_exp"].rolling(4, min_periods=0).mean().fillna(0)
        sum_exp = (oper_cost_TTM + fin_exp_TTM + sell_exp_TTM + admin_exp_TTM)
        total_profit_cost_ratio_TTM = total_profit_TTM / sum_exp
        total_profit_cost_ratio = self.drop_duplicated_ann_date(total_profit_cost_ratio_TTM)
        TotalProfitCostRatio = self.expand_index(total_profit_cost_ratio, self.df_sb)
        return TotalProfitCostRatio

    def ROA(self):
        """
        因子描述：
        资产回报率（Return on assets）。
        计算方法：
        资产回报率 = 归属于母公司所有者的净利润TTM / 最新四个季度总资产平均值
        注：分母<0时，因子值为空
        """
        NIAP = self.df_bic["n_income_attr_p"].rolling(4, 0).mean()
        to_asset = self.df_bic["total_assets"]
        t_asset = to_asset.rolling(window=4, min_periods=0).mean()
        at_asset = t_asset.mask(t_asset < 0, np.nan)
        roa = NIAP / at_asset
        t = self.drop_duplicated_ann_date(roa)
        ROA = self.expand_index(t, self.df_sb)
        return ROA

    def ROA5(self):
        """
        因子描述：
        5 年资产回报率（Five-year average return on assets）。
        计算方法：
        5 年资产回报率 = 归属于母公司所有者的净利润TTM / 最近5年平均总资产
        取最近五年内的年报数据，分母的年数应比分子年向前错一年，最后结果仅保留记录数不小于三年的数据。
        符号说明：
        符号 描述 计算方法
        NPAP 归属于母公司所有者的净利润 TTM
        TA 总资产 Mean
        注：分母<0时，因子值为空
        """
        NIAP = self.df_bic["n_income_attr_p"].rolling(4, 0).mean()
        lag_to_asset = self.df_bic["total_assets"].shift(4)
        lag_t_asset = lag_to_asset.rolling(window=20, min_periods=12).mean()
        lag_at_asset = lag_t_asset.mask(lag_t_asset < 0, np.nan)
        roa5 = NIAP / lag_at_asset
        t = self.drop_duplicated_ann_date(roa5)
        ROA5 = self.expand_index(t, self.df_sb)
        return ROA5

    def ROE(self):
        """
        因子描述：
        权益回报率（Return on equity）。
        计算方法：
        权益回报率 = 归属于母公司所有者的净利润TTM / 最近四个季度归属于母公司所有者权益的平均值
        注：分母<0时，因子值为空
        """
        NIAP = self.df_bic["n_income_attr_p"].rolling(4, 0).mean()
        to_Equity = self.df_bic["total_hldr_eqy_exc_min_int"].fillna(0)
        to_Equity = to_Equity.rolling(4, 0).mean()
        at_Equity = to_Equity.mask(to_Equity < 0, np.nan)
        roe = NIAP / at_Equity
        t = self.drop_duplicated_ann_date(roe)
        ROE = self.expand_index(t, self.df_sb)
        return ROE

    def ROE5(self):
        """
        因子描述：
        5 年权益回报率（Five-year average return on equity）。
        计算方法：
        5 年权益回报率 = 归属于母公司所有者的净利润TTM / 归属于母公司所有者权益
        取最近五年内的年报数据，分母的年数应比分子年向前错一年，最后结果仅保留记录数不小于三年的数据。
        符号说明：
        符号 描述 计算方法
        NPAP 归属于母公司所有者的净利润 TTM
        TSEP 归属于母公司所有者权益 Mean
        注：分母<0时，因子值为空
        """
        NIAP = self.df_bic["n_income_attr_p"].rolling(window=4, min_periods=0).mean()
        lag_to_Equity = self.df_bic["total_hldr_eqy_exc_min_int"].shift(4)
        lag_to_Equity = lag_to_Equity.rolling(window=4, min_periods=0).mean()
        lag_at_Equity = lag_to_Equity.mask(lag_to_Equity < 0, np.nan)
        roe5 = NIAP / lag_at_Equity
        t = self.drop_duplicated_ann_date(roe5)
        ROE5 = self.expand_index(t, self.df_sb)
        return ROE5

    def EGRO(self):
        """
        因子描述：
        5 年收益增长率（Five-year earnings growth）。
        计算方法：
        5 年收益增长率 = 5年收益关于时间（年）进行线性回归的回归系数 / 5年收益均值的绝对值
        其中NP取年报数据，t 是时间，β 是5 年收益关于时间（年）进行线性回归的回归系数，mean(NP) 是5 年收益均值。进行线性回归时，
        当数据库中记录数大于五年时取最近五年的数据，记录数不足五年时取全部数据，最后结果仅保留记录数不小于三年的数据。
        """
        NP = self.df_bic['net_profit']

        def temp(Y):
            array = np.array([5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
            mod = sm.OLS(Y, sm.add_constant(array))
            res = mod.fit()
            return res.params[1]

        b = NP.rolling(window=20, min_periods=12).apply(temp)
        EGRO = b / abs(NP.rolling(window=20, min_periods=12).mean())
        EGRO = self.drop_duplicated_ann_date(EGRO)
        EGRO = self.expand_index(EGRO, self.df_sb)
        return EGRO

    def SUE(self):
        """
        # TODO
        因子描述：
        未预期盈余（Standardized unexpected earnings）。
        计算方法：
        未预期盈余 = ( 最近一年净利润 - 除去最近一年的过往净利润均值 ) / 除去最近一年的过往净利润标准差
        当数据库中记录数大于五年时取最近五年的数据，记录数不足五年时取全部数据，最后结果仅保留记录数不小于三年的数据。其中净利润按时间排序，年份由近至远，例如2014，2013，2012，……。
         为最近一年的数据， 为除去最近一年数据的过往数据均值， 为除去最近一年数据的过往数据标准差。
        符号 描述 计算方法
        NP 净利润 TTM
        """
        net_profit_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        SUE = (net_profit_TTM.rolling(window=4, min_periods=0).mean() -
               net_profit_TTM.shift(4).rolling(window=16, min_periods=8).mean()) / \
              net_profit_TTM.shift(4).rolling(window=16, min_periods=8).std()
        SUE = self.drop_duplicated_ann_date(SUE)
        SUE = self.expand_index(SUE, self.df_sb)
        return SUE

    def SUOI(self):
        """
        # TODO
        因子描述：
        未预期毛利（Standardized unexpected gross income）。
        计算方法：
        －
        其中OR是营业收入TTM值，OC是营业成本TTM值。
        当数据库中记录数大于五年时取最近五年的数据，记录数不足五年时取全部数据，最后结果仅保留记录数不小于三年的数
        据。其中净利润按时间排序，年份由近至远，例如2014，2013，2012，……。
         为最近一年的数据， 为除去最近一年数据的过往数据均值， 为除去最近一年数据的过往
        数据标准差。
        """
        OR = self.df_bic['revenue'].rolling(window=4, min_periods=0).mean()
        OC = self.df_bic['oper_cost'].rolling(window=4, min_periods=0).mean()
        GI = OR - OC
        SUOI = (GI.rolling(window=4, min_periods=0).mean() - GI.shift(4).rolling(window=16, min_periods=8).mean()) / \
               GI.shift(4).rolling(window=16, min_periods=8).std()
        SUOI = self.drop_duplicated_ann_date(SUOI)
        SUOI = self.expand_index(SUOI, self.df_sb)
        return SUOI

    def ETOP(self):
        """
        因子描述：
        收益市值比（Earnings to price）。
        计算方法：
        收益市值比 = 净利润(TTM, income) / 总市值(Latest, stock_basic)。
        """
        net_profit_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        net_profit = self.drop_duplicated_ann_date(net_profit_TTM)
        NP = self.expand_index(net_profit, self.df_sb)
        total_mv = self.df_sb["total_mv"]
        ETOP = NP / total_mv
        return ETOP

    def ETP5(self):
        """
        因子描述：
        5年平均收益市值比（Five-year average earnings to price）。
        计算方法：
        5年平均收益市值比 = 近五年总净利润 / 近五年总市值。
        净利润取年报数据。
        """
        t_net_profit = self.df_bic["net_profit"].rolling(window=20, min_periods=0).sum()
        net_profit = self.drop_duplicated_ann_date(t_net_profit)
        NP = self.expand_index(net_profit, self.df_sb)
        total_mv = self.df_sb["total_mv"].rolling(window=1250, min_periods=1).mean()
        ETP5 = NP / (total_mv * 5)
        return ETP5

    def NetNonOIToTP(self):
        """
        因子描述：
        营业外收支净额(TTM)/利润总额(TTM)。
        计算方法：
        营业外收支净额/利润总额 = （ 营业外收入(TTM, income) - 营业外支出(TTM, income) ）/利润总额(TTM, income)
        注：当企业利润为负，但营业外是赚钱的时候会导致该因子值为负。
        """
        non_oper_income_TTM = self.df_bic["non_oper_income"].rolling(4, min_periods=0).mean().fillna(0)
        non_oper_exp_TTM = self.df_bic["non_oper_exp"].rolling(4, min_periods=0).mean().fillna(0)
        total_profit_TTM = self.df_bic["total_profit"].rolling(4, min_periods=0).mean()
        non_oper_diff_TTM = non_oper_income_TTM - non_oper_exp_TTM
        net_non_oi_to_tp_TTM = non_oper_diff_TTM / total_profit_TTM
        net_non_oi_to_tp = self.drop_duplicated_ann_date(net_non_oi_to_tp_TTM)
        NetNonOIToTP = self.expand_index(net_non_oi_to_tp, self.df_sb)
        return NetNonOIToTP

    def NetNonOIToTPLatest(self):
        """
        因子描述：
        营业外收支净额/利润总额。
        计算方法：
        营业外收支净额/利润总额 = （ 营业外收入(Latest, income)-营业外支出(Latest, income) ） / 利润总额(Latest, income)
        """
        non_oper_income = self.df_bic["non_oper_income"].fillna(0)
        non_oper_exp = self.df_bic["non_oper_exp"].fillna(0)
        total_profit = self.df_bic["total_profit"]
        non_oper_diff = non_oper_income - non_oper_exp
        net_non_oi_to_tp_latest = non_oper_diff / total_profit
        net_non_oi_to_tp = self.drop_duplicated_ann_date(net_non_oi_to_tp_latest)
        NetNonOIToTPLatest = self.expand_index(net_non_oi_to_tp, self.df_sb)
        return NetNonOIToTPLatest

    def PeriodCostsRate(self):
        """
        因子描述：
        销售期间费用率。
        计算方法：
        销售期间费用率 = ( 财务费用(TTM, income) + 销售费用(TTM, income) + 管理费用(TTM, income) ) / 营业收入(TTM, income)
        注：在企业利润为负时，营业收入很低，同时三费很高，所以可能导致该因子值大于1。
        """
        fin_exp_TTM = self.df_bic["fin_exp"].rolling(4, min_periods=0).mean().fillna(0)
        sell_exp_TTM = self.df_bic["sell_exp"].rolling(4, min_periods=0).mean().fillna(0)
        admin_exp_TTM = self.df_bic["admin_exp"].rolling(4, min_periods=0).mean().fillna(0)
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        sum_exp = (fin_exp_TTM + sell_exp_TTM + admin_exp_TTM)
        period_costs_rate_TTM = sum_exp / revenue_TTM
        period_costs_rate = self.drop_duplicated_ann_date(period_costs_rate_TTM)
        PeriodCostsRate = self.expand_index(period_costs_rate, self.df_sb)
        return PeriodCostsRate

    def ROEDiluted(self):
        """
        # TODO
        因子描述：
        净资产收益率（摊薄）。
        计算方法：
        净资产收益率（摊薄） = 归属于母公司的净利润 / 期末归属于母公司的所有者权益 * 100%
        """
        ROE_Diluted = self.df_bic['n_income_attr_p'] / self.df_bic['total_hldr_eqy_exc_min_int']
        ROE_Diluted = self.drop_duplicated_ann_date(ROE_Diluted)
        ROEDiluted = self.expand_index(ROE_Diluted, self.df_sb)
        return ROEDiluted

    def ROEAvg(self):
        """
        # TODO
        因子描述：
        净资产收益率（平均）。
        计算方法：
        净资产收益率（平均） = 归属于母公司的净利润 * 2 /（期末归属于母公司的所有者权益 + 期初归属于股公司的所有者权益）*100%
        """

    def ROEWeighted(self):
        """
        # TODO
        因子描述：
        净资产收益率（加权平均，公布值）。
        计算方法：
        直接取公告披露值
        """

    def ROECut(self):
        """
        # TODO
        因子描述：
        净资产收益率（扣除摊薄）。
        计算方法：
        净资产收益率（扣除摊薄） = 扣非归属于母公司的净利润 / 期末归属于母公司的所有者权益
        附注：扣非归属于母公司的净利润取公司披露值，若未披露则：归属于母公司净利润-非经常性损益
        """
        E_I = self.df_bic['non_oper_income'] - self.df_bic['non_oper_exp']
        tmp = self.df_bic['n_income_attr_p'] - E_I
        ROE_Cut = tmp / self.df_bic['total_hldr_eqy_exc_min_int']
        ROE_Cut = self.drop_duplicated_ann_date(ROE_Cut)
        ROECut = self.expand_index(ROE_Cut, self.df_sb)
        return ROECut

    def ROECutWeighted(self):
        """
        # TODO
        因子描述：
        净资产收益率（扣除加权平均，公布值）。
        计算方法：
        直接取公告披露值
        """

    def ROIC(self):
        """
        # TODO
        因子描述：
        投入资本回报率。
        计算方法：
        投入资本回报率 = 息前税后利润 * 2 / (期初投入资本+期末投入资本)
        """

    def ROAEBIT(self):
        """

        因子描述：
        总资产报酬率。
        计算方法：
        总资产报酬率 = 息税前利润 * 2 / (期初总资产+期末总资产)
        ebit
        """

    def ROAEBITTTM(self):
        """
        # TODO
        因子描述：
        总资产报酬率（TTM）。
        计算方法：
        总资产报酬率（TTM） = 息税前利润TTM / 总资产
        总资产去除当季度后，取最新四季度平均值。
        注：分母<0时，因子值为空
        """
        LA = self.df_bic['total_assets']
        LA = LA.fillna(method="pad")
        ROAEBIT_TTM = self.df_bic['ebit'].rolling(4, min_periods=0).mean() / LA.shift(1).rolling(4,
                                                                                                 min_periods=0).mean()
        ROAEBIT_TTM = self.drop_duplicated_ann_date(ROAEBIT_TTM)
        ROAEBITTTM = self.expand_index(ROAEBIT_TTM, self.df_sb)
        return ROAEBITTTM

    def OperatingNIToTP(self):
        """
        # TODO
        因子描述：
        经营活动净收益/利润总额。
        计算方式：
        经营活动净收益/利润总额 = 经营活动净收益TTM /利润总额TTM
        注：分母<0时，因子值为空
        """
        NIFO = self.df_bic['total_revenue'] - self.df_bic['total_cogs']
        Operating_NIToTP = NIFO.rolling(4, min_periods=0).mean() / \
                           self.df_bic['total_profit'].rolling(4, min_periods=0).mean()
        Operating_NIToTP = self.drop_duplicated_ann_date(Operating_NIToTP)
        OperatingNIToTP = self.expand_index(Operating_NIToTP, self.df_sb)
        return OperatingNIToTP

    def OperatingNIToTPLatest(self):
        """
        # TODO
        因子描述：
        经营活动净收益/利润总额（Latest）。
        计算方式：
        经营活动净收益/利润总额 = 经营活动净收益 / 利润总额（Latest）
        """
        NIFO = self.df_bic['total_revenue'] - self.df_bic['total_cogs']
        Operating_NIToTPLatest = NIFO / self.df_bic['total_profit']
        Operating_NIToTPLatest = self.drop_duplicated_ann_date(Operating_NIToTPLatest)
        OperatingNIToTPLatest = self.expand_index(Operating_NIToTPLatest, self.df_sb)
        return OperatingNIToTPLatest

    def InvestRAssociatesToTP(self):
        """
        # TODO
        因子描述：
        对联营和营公司投资收益/利润总额。
        计算方法：
        对联营和营公司投资收益/利润总额 = 对联营和营公司投资收益TTM / 利润总额TTM
        注：分母<0时，因子值为空
        """
        Invest_RAssociatesToTP = self.df_bic['ass_invest_income'].rolling(4, min_periods=0).mean() / \
                                 self.df_bic['total_profit'].rolling(4, min_periods=0).mean()
        Invest_RAssociatesToTP = self.drop_duplicated_ann_date(Invest_RAssociatesToTP)
        InvestRAssociatesToTP = self.expand_index(Invest_RAssociatesToTP, self.df_sb)
        return InvestRAssociatesToTP

    def InvestRAssociatesToTPLatest(self):
        """
        # TODO
        因子描述：
        对联营和营公司投资收益/利润总额（Latest）。
        计算方法：
        对联营和营公司投资收益/利润总额 = 对联营和营公司投资收益 / 利润总额
        财务科目的空值用前值填充
        """
        Invest_RAssociatesToTPLatest = self.df_bic['ass_invest_income'] / self.df_bic['total_profit']
        Invest_RAssociatesToTPLatest = self.drop_duplicated_ann_date(Invest_RAssociatesToTPLatest)
        InvestRAssociatesToTPLatest = self.expand_index(Invest_RAssociatesToTPLatest, self.df_sb)
        return InvestRAssociatesToTPLatest

    def NPCutToNP(self):
        """
        # TODO
        因子描述：
        扣除非经常损益后的净利润/净利润(Latest)。
        计算方法：
        扣除非经常损益后的净利润/净利润 = 扣除非经常损益后的净利润 / 净利润(Latest)
        财务科目的空值用前值填充
        """
        E_I = self.df_bic['non_oper_income'] - self.df_bic['non_oper_exp']
        NPCutToNP_tmp = 1 - (E_I / self.df_bic['net_profit'])
        NPCutToNP_tmp = self.drop_duplicated_ann_date(NPCutToNP_tmp)
        NPCutToNP = self.expand_index(NPCutToNP_tmp, self.df_sb)
        return NPCutToNP

