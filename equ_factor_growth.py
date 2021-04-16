import logging
import pandas as pd
import numpy as np

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_成长能力 15 - 13
class EquFactorGrowth(Equ):

    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb

    def NetAssetGrowRate(self):
        """
        因子描述：
        净资产增长率（Net assets growth rate）。
        计算方法：
        净资产增长率 = ( 今年股东权益(Latest, balancesheet) - 去年股东权益 ) / abs( 去年股东权益 )
        注：分母<0时，因子值为空
        """
        total_hldr = self.df_bic["total_hldr_eqy_inc_min_int"]
        last_total_hldr = total_hldr.shift(periods=4, fill_value=0)
        equity_diff = (total_hldr - last_total_hldr) / (last_total_hldr.abs())
        net_asset_grow_rate = self.drop_duplicated_ann_date(equity_diff)
        NetAssetGrowRate = self.expand_index(net_asset_grow_rate, self.df_sb)
        return NetAssetGrowRate

    def TotalAssetGrowRate(self):
        """
        因子描述：
        总资产增长率（Total assets growth rate）。
        计算方法：
        总资产增长率 = ( 今年总资产(Latest, balancesheet) - 去年总资产 ) / abs( 去年总资产 )
        注：分母<0时，因子值为空
        """
        total_assets = self.df_bic["total_assets"]
        last_total_assets = total_assets.shift(periods=4, fill_value=0)
        assets_diff = (total_assets - last_total_assets) / (last_total_assets.abs())
        total_asset_grow_rate = self.drop_duplicated_ann_date(assets_diff)
        TotalAssetGrowRate = self.expand_index(total_asset_grow_rate, self.df_sb)
        return TotalAssetGrowRate

    def OperatingRevenueGrowRate(self):
        """
        因子描述：
        营业收入增长率（Operating revenue growth rate）。
        计算方法：
        营业收入增长率 = ( 今年营业收入(TTM, income) - 去年营业收入TTM ) / abs( 去年营业收入TTM )
        注：分母<0时，因子值为空
        """
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        last_revenue_TTM = revenue_TTM.shift(periods=4, fill_value=0)
        revenue_TTM_diff = (revenue_TTM - last_revenue_TTM) / (last_revenue_TTM.abs())
        operating_revenue_grow_rate = self.drop_duplicated_ann_date(revenue_TTM_diff)
        OperatingRevenueGrowRate = self.expand_index(operating_revenue_grow_rate, self.df_sb)
        return OperatingRevenueGrowRate

    def OperatingProfitGrowRate(self):
        """
        因子描述：
        营业利润增长率（Operating profit growth rate）。
        计算方法：
        营业利润增长率 = ( 今年营业利润(TTM, income) - 去年营业利润TTM ) / abs( 去年营业利润TTM )
        """
        operate_profit_TTM = self.df_bic["operate_profit"].rolling(4, min_periods=0).mean()
        last_operate_profit_TTM = operate_profit_TTM.shift(periods=4, fill_value=0)
        operate_profit_TTM_diff = (operate_profit_TTM - last_operate_profit_TTM) / (last_operate_profit_TTM.abs())
        operating_profit_grow_rate = self.drop_duplicated_ann_date(operate_profit_TTM_diff)
        OperatingProfitGrowRate = self.expand_index(operating_profit_grow_rate, self.df_sb)
        return OperatingProfitGrowRate

    def TotalProfitGrowRate(self):
        """
        因子描述：
        利润总额增长率（Total profit growth rate）。
        计算方法：
        利润总额增长率 = ( 今年利润总额(TTM, income) - 去年利润总额TTM ) / abs( 去年利润总额TTM )
        """
        total_profit_TTM = self.df_bic["total_profit"].rolling(4, min_periods=0).mean()
        last_total_profit_TTM = total_profit_TTM.shift(periods=4, fill_value=0)
        total_profit_TTM_diff = (total_profit_TTM - last_total_profit_TTM) / (last_total_profit_TTM.abs())
        total_profit_grow_rate = self.drop_duplicated_ann_date(total_profit_TTM_diff)
        TotalProfitGrowRate = self.expand_index(total_profit_grow_rate, self.df_sb)
        return TotalProfitGrowRate

    def NetProfitGrowRate(self):
        """
        因子描述：
        净利润增长率（Net profit growth rate）。
        计算方法：
        净利润增长率 = ( 今年净利润(TTM, income) - 去年净利润TTM ) / abs( 去年净利润TTM )
        """
        income_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        last_income_TTM = income_TTM.shift(periods=4, fill_value=0)
        income_TTM_diff = (income_TTM - last_income_TTM) / (last_income_TTM.abs())
        net_profit_grow_rate = self.drop_duplicated_ann_date(income_TTM_diff)
        NetProfitGrowRate = self.expand_index(net_profit_grow_rate, self.df_sb)
        return NetProfitGrowRate

    def NPParentCompanyGrowRate(self):
        """
        因子描述：
        归属母公司股东的净利润增长率（Growth rate of net income attributable to shareholders of parent company）。
        计算方法：
        归属母公司股东的净利润增长率 = ( 今年归属于母公司所有者的净利润(TTM, income) - 去年归属于母公司所有者的净利润TTM ) / abs( 去年归属于母公司所有者的净利润TTM )
        """
        income_attr_p_TTM = self.df_bic["n_income_attr_p"].rolling(4, min_periods=0).mean()
        last_income_attr_p_TTM = income_attr_p_TTM.shift(periods=4, fill_value=0)
        income_attr_p_TTM_diff = (income_attr_p_TTM - last_income_attr_p_TTM) / (last_income_attr_p_TTM.abs())
        NP_parent_company_grow_rate = self.drop_duplicated_ann_date(income_attr_p_TTM_diff)
        NPParentCompanyGrowRate = self.expand_index(NP_parent_company_grow_rate, self.df_sb)
        return NPParentCompanyGrowRate

    def DEGM(self):
        """
        因子描述：
        毛利率增长（Growth rate of gross income ratio），去年同期相比。
        计算方法：
        毛利率增长 = [ ( 当期营业收入(TTM, income) - 当期营业成本(TTM, income) ) / 当期营业收入 ]
                - [ ( 去年同期营业收入 - 去年同期营业成本 ) / 去年同期营业收入 ]
        """
        OR_TTM = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        OC_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        last_OR_TTM = OR_TTM.shift(periods=4, fill_value=0)
        last_OC_TTM = OC_TTM.shift(periods=4, fill_value=0)
        OR_OC_TTM = (OR_TTM - OC_TTM) / OR_TTM
        last_OR_OC_TTM = (last_OR_TTM - last_OC_TTM) / last_OR_TTM
        degm_TTM = OR_OC_TTM.sub(last_OR_OC_TTM, fill_value=0)
        degm = self.drop_duplicated_ann_date(degm_TTM)
        DEGM = self.expand_index(degm, self.df_sb)
        return DEGM

    def EARNMOM(self):
        """
        因子描述：
        八季度净利润变化趋势（Growth tendency of net profit in the past eight quarters），前8个季度的净利润，如果同
        比（去年同期）增长记为+1，同比下滑记为-1，再将8个值相加。
        计算方法：
        八季度净利润变化趋势 = sign(7年前净利润(TTM, income) - 8年前净利润)+sign(6年前净利润 - 7年前净利润)+...+sign(当期净利润 - 上一年净利润)
        """
        NP = self.df_bic["net_profit"].rolling(4).mean().fillna(0)
        NP_lag = NP.shift(4)
        diff = NP - NP_lag
        sign = 2*(diff > 0) - 1
        EARNMOM = sign.rolling(8).sum().fillna(0)
        return EARNMOM

    def NetProfitGrowRate3Y(self):
        """
        因子描述：
        净利润3年复合增长率
        计算方法：
        净利润3年复合增长率 = sign( 当期净利润(TTM, income) ) * ( abs( 当期净利润 / 3年前净利润 ) ^ ( 1 / 3 ) ) - 1
        """
        NP_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        NP_3_TTM = NP_TTM.shift(periods=12, fill_value=0)
        net_profit_grow_rate = np.sign(NP_TTM) * ((NP_TTM / NP_3_TTM).abs().pow(1 / 3, fill_value=0)) - 1
        net_profit_grow_rate_3Y = self.drop_duplicated_ann_date(net_profit_grow_rate)
        NetProfitGrowRate3Y = self.expand_index(net_profit_grow_rate_3Y, self.df_sb)
        return NetProfitGrowRate3Y

    def NetProfitGrowRate5Y(self):
        """
        因子描述：
        净利润5年复合增长率
        计算方法：
        净利润5年复合增长率 = sign( 当期净利润(TTM, income) ) * ( abs( 当期净利润 / 5年前净利润 ) ^ ( 1 / 5 ) ) - 1
        """
        NP_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        NP_5_TTM = NP_TTM.shift(periods=20, fill_value=0)
        net_profit_grow_rate = np.sign(NP_TTM) * ((NP_TTM / NP_5_TTM).abs().pow(1 / 5, fill_value=0)) - 1
        net_profit_grow_rate_5Y = self.drop_duplicated_ann_date(net_profit_grow_rate)
        NetProfitGrowRate5Y = self.expand_index(net_profit_grow_rate_5Y, self.df_sb)
        return NetProfitGrowRate5Y

    def OperatingRevenueGrowRate3Y(self):
        """
        因子描述：
        营业收入3年复合增长率
        计算方法：
        营业收入3年复合增长率 = sign( 当期营业收入(TTM, income) ) * ( abs( 当期营业收入 / 3年前营业收入 ) ^ ( 1 / 3 ) ) - 1
        """
        OR_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        OR_3_TTM = OR_TTM.shift(periods=12, fill_value=0)
        operating_revenue_grow_rate = np.sign(OR_TTM) * ((OR_TTM / OR_3_TTM).abs().pow(1 / 3, fill_value=0)) - 1
        operating_revenue_grow_rate_3Y = self.drop_duplicated_ann_date(operating_revenue_grow_rate)
        OperatingRevenueGrowRate3Y = self.expand_index(operating_revenue_grow_rate_3Y, self.df_sb)
        return OperatingRevenueGrowRate3Y

    def OperatingRevenueGrowRate5Y(self):
        """
        因子描述：
        营业收入5年复合增长率
        计算方法：
        营业收入5年复合增长率 = sign( 当期营业收入(TTM, income) ) * ( abs( 当期营业收入 / 5年前营业收入 ) ^ ( 1 / 5 ) ) - 1
        """
        OR_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        OR_5_TTM = OR_TTM.shift(periods=20, fill_value=0)
        operating_revenue_grow_rate = np.sign(OR_TTM) * ((OR_TTM / OR_5_TTM).abs().pow(1 / 5, fill_value=0)) - 1
        operating_revenue_grow_rate_5Y = self.drop_duplicated_ann_date(operating_revenue_grow_rate)
        OperatingRevenueGrowRate5Y = self.expand_index(operating_revenue_grow_rate_5Y, self.df_sb)
        return OperatingRevenueGrowRate5Y

    def NetCashFlowGrowRate(self):
        """
        因子描述：
        净现金流量增长率
        计算方式：
        净现金流量增长率 = ( 当期现金及现金等价物净增加额(TTM, cashflow) - 上一年现金及现金等价物净增加额 )
                        / abs( 上一年现金及现金等价物净增加额 )
        """
        NCC_TTM = self.df_bic["n_incr_cash_cash_equ"].rolling(4, min_periods=0).mean()
        last_NCC_TTM = NCC_TTM.shift(periods=4, fill_value=0)
        NCC_TTM_diff = (NCC_TTM - last_NCC_TTM) / (last_NCC_TTM.abs())
        net_cashflow_grow_rate = self.drop_duplicated_ann_date(NCC_TTM_diff)
        NetCashFlowGrowRate = self.expand_index(net_cashflow_grow_rate, self.df_sb)
        return NetCashFlowGrowRate

    def NPParentCompanyCutYOY(self):
        """
        因子描述：
        归属母公司股东的净利润(扣除非经常损益)同比增长（%）。
        计算方法：
        归属母公司股东的净利润(扣除非经常损益)同比增长 = ( 当期归属母公司股东的净利润(扣除)(TTM) - 上一年归属母公司股东的净利润(扣除) )
                                                / abs( 上一年归属母公司股东的净利润(扣除) )
        非经常性损益 = 营业外收入-营业外支出
        """
        NIAP = self.df_bic["n_income_attr_p"]
        NOI = self.df_bic["non_oper_income"]
        NOE = self.df_bic["non_oper_exp"]
        ex_U = NOI - NOE
        NPP = NIAP - ex_U
        NPP_TTM = NPP.rolling(4,min_periods=0).mean().fillna(0)
        last_NPP_TTM = NPP_TTM.shift(periods=4,fill_value=0)
        NPP_YOY = (NPP_TTM - last_NPP_TTM) / abs(last_NPP_TTM)
        NPP_yoy = self.drop_duplicated_ann_date(NPP_YOY)
        NPParentCompanyCutYOY = self.expand_index(NPP_yoy, self.df_sb)
        return NPParentCompanyCutYOY

