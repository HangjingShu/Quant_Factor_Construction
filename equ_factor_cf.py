import numpy as np
import pandas as pd

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_现金流指标 17
class EquFactorCf(Equ):
    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb

    def FinancingCashGrowRate(self):
        """
        因子描述：
        筹资活动产生的现金流量净额增长率（Growth rate of financing cash flow）。
        计算方法：
        筹资活动产生的现金流量净额增长率 = ( 今年筹资活动产生的现金流量净额(TTM, income) - 去年筹资活动产生的现金流量净额TTM )
                                    / abs( 去年筹资活动产生的现金流量净额TTM )
        """
        fnc_act_TTM = self.df_bic["n_cash_flows_fnc_act"].rolling(4, min_periods=0).mean().fillna(0)
        last_fnc_act_TTM = fnc_act_TTM.shift(periods=4, fill_value=0)
        fnc_act_TTM_diff = (fnc_act_TTM - last_fnc_act_TTM) / (last_fnc_act_TTM.abs())
        financing_cash_grow_rate = self.drop_duplicated_ann_date(fnc_act_TTM_diff)
        FinancingCashGrowRate = self.expand_index(financing_cash_grow_rate, self.df_sb)
        return FinancingCashGrowRate

    def OperCashGrowRate(self):
        """
        因子描述：
        经营活动产生的现金流量净额增长率（Growth rate of operating cash flow）。
        计算方法：
        经营活动产生的现金流量净额增长率 = ( 今年经营活动产生的现金流量净额(TTM, cashflow) - 去年经营活动产生的现金流量净额TTM )
                                    / abs( 去年经营活动产生的现金流量净额TTM )
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean().fillna(0)
        last_act_TTM = act_TTM.shift(periods=4, fill_value=0)
        act_TTM_diff = (act_TTM - last_act_TTM) / (last_act_TTM.abs())
        oper_cash_grow_rate = self.drop_duplicated_ann_date(act_TTM_diff)
        OperCashGrowRate = self.expand_index(oper_cash_grow_rate, self.df_sb)
        return OperCashGrowRate

    def InvestCashGrowRate(self):
        """
        因子描述：
        投资活动产生的现金流量净额增长率（Growth rate of investing cash flow）。
        计算方法：
        投资活动产生的现金流量净额增长率 = ( 今年投资活动产生的现金流量净额(TTM, cashflow) - 去年投资活动产生的现金流量净额TTM )
                                    / abs( 去年投资活动产生的现金流量净额TTM )
        """
        inv_act_TTM = self.df_bic["n_cashflow_inv_act"].rolling(4, min_periods=0).mean().fillna(0)
        last_inv_act_TTM = inv_act_TTM.shift(periods=4, fill_value=0)
        inv_act_TTM_diff = (inv_act_TTM - last_inv_act_TTM) / (last_inv_act_TTM.abs())
        invest_cash_grow_rate = self.drop_duplicated_ann_date(inv_act_TTM_diff)
        InvestCashGrowRate = self.expand_index(invest_cash_grow_rate, self.df_sb)
        return InvestCashGrowRate

    def SaleServiceCashToOR(self):
        """
        因子描述：
        销售商品提供劳务收到的现金与营业收入之比（Sale service cash to operating revenues）
        计算方法：
        销售商品提供劳务收到的现金与营业收入之比 = 销售商品和提供劳务收到的现金(TTM, cashflow) / 营业收入(TTM, income)
        """
        sale_sg_TTM = self.df_bic["c_fr_sale_sg"].rolling(4, min_periods=0).mean()
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        sale_sg_revenue_TTM = sale_sg_TTM / revenue_TTM
        sale_service_cash_to_OR = self.drop_duplicated_ann_date(sale_sg_revenue_TTM)
        SaleServiceCashToOR = self.expand_index(sale_service_cash_to_OR, self.df_sb)
        return SaleServiceCashToOR

    def CashRateOfSales(self):
        """
        因子描述：
        经营活动产生的现金流量净额与营业收入之比（operating cash flow-to-Revenue ratio）
        计算方法：
        经营活动产生的现金流量净额与营业收入之比 = 经营活动产生的现金流量净额(TTM, cashflow) / 营业收入(TTM, income)
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        revenue_TTM = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        act_revenue_TTM = act_TTM / revenue_TTM
        cash_rate_of_sales = self.drop_duplicated_ann_date(act_revenue_TTM)
        CashRateOfSales = self.expand_index(cash_rate_of_sales, self.df_sb)
        return CashRateOfSales

    def NOCFToOperatingNI(self):
        """
        因子描述：
        经营活动产生的现金流量净额与经营活动净收益之比（Opearting cash flow-to-net income ratio）
        计算方法：
        经营活动产生的现金流量净额与经营活动净收益之比 = 经营活动产生的现金流量净额(TTM, cashflow)
                                                / ( 营业总收入(TTM, income) - 营业总成本(TTM, income) )
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        total_revenue_TTM = self.df_bic["total_revenue"].rolling(4, min_periods=0).mean()
        total_cogs_TTM = self.df_bic["total_cogs"].rolling(4, min_periods=0).mean()
        total_TTM = act_TTM / (total_revenue_TTM - total_cogs_TTM)
        NOCF_to_operating_NI = self.drop_duplicated_ann_date(total_TTM)
        NOCFToOperatingNI = self.expand_index(NOCF_to_operating_NI, self.df_sb)
        return NOCFToOperatingNI

    def OperCashInToCurrentLiability(self):
        """
        因子描述：
        现金流动负债比（Opearting cash flow-to-current liability ratio）
        计算方法：
        现金流动负债比 = 经营活动产生的现金流量净额(TTM, cashflow) / 流动负债合计(balancesheet)
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        total_cur_liab = self.df_bic["total_cur_liab"]
        act_total_cur_liab = act_TTM / total_cur_liab
        oper_cash_in_to_current_liability = self.drop_duplicated_ann_date(act_total_cur_liab)
        OperCashInToCurrentLiability = self.expand_index(oper_cash_in_to_current_liability, self.df_sb)
        return OperCashInToCurrentLiability

    def CashToCurrentLiability(self):
        """
        因子描述：
        现金比率（Cash Ratio ）
        计算方法：
        现金比率 = 期末现金及现金等价物余额(TTM, cashflow) / 流动负债合计(balancesheet)
        """
        equ_end_period_TTM = self.df_bic["c_cash_equ_end_period"].rolling(4, min_periods=0).mean()
        total_cur_liab = self.df_bic["total_cur_liab"]
        end_total_cur_liab = equ_end_period_TTM / total_cur_liab
        cash_to_current_liability = self.drop_duplicated_ann_date(end_total_cur_liab)
        CashToCurrentLiability = self.expand_index(cash_to_current_liability, self.df_sb)
        return CashToCurrentLiability

    def CTOP(self):
        """
        因子描述：
        现金流市值比（Cash flow to price）
        计算方法：
        现金流市值比 = 每股派现（税前） × 分红前总股本 / 总市值
        注： 分子为最近4期财报对应的派现总和，只要发了公告就考虑在分子中；分母市值为因子当日的总市值；
        """

    def CTP5(self):
        """
        因子描述：
        5 年平均现金流市值比（Five-year average cash flow to price）
        计算方法：
        5 年平均现金流市值比 = 近五年每股派现（税前） × 分红前总股本 / 近五年总市值
        近五年每股派现（税前）取年报数据
        注：分母中，总市值取的是每年最后一个交易日的市值；分子仅考虑已经除权除息的分红，如果发了公告但未实施的，不予考虑
        """

    def CFO2EV(self):
        """
        因子描述：
        经营活动产生的现金流量净额与企业价值之比（Operating cash flow to enterprise value）
        计算方法：
        经营活动产生的现金流量净额与企业价值之比 = 经营活动产生的现金流量净额(TTM, cashflow)
                                            / (长期借款(Latest, balancesheet) + 短期借款(Latest, balancesheet)
                                               + 总市值(Latest) - 现金及现金等价物(Latest, cashflow))
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        lt_borr = self.df_bic["lt_borr"].fillna(0)
        st_borr = self.df_bic["st_borr"].fillna(0)
        equ_end_period = self.df_bic["c_cash_equ_end_period"].fillna(0)
        ACT = self.expand_index(self.drop_duplicated_ann_date(act_TTM), self.df_sb)
        LT = self.expand_index(self.drop_duplicated_ann_date(lt_borr), self.df_sb)
        ST = self.expand_index(self.drop_duplicated_ann_date(st_borr), self.df_sb)
        EquEnd = self.expand_index(self.drop_duplicated_ann_date(equ_end_period), self.df_sb)
        total_mv = self.df_sb["total_mv"]
        CFO2EV = ACT / (LT + ST + total_mv - EquEnd)
        return CFO2EV

    def ACCA(self):
        """
        因子描述：
        现金流资产比和资产回报率之差（Cash flow assets ratio minus return on assets）
        计算方法：
        现金流资产比和资产回报率之差 = ( 经营活动产生的现金流量净额(TTM, cashflow) - 净利润(TTM, income) )
                                / 总资产(Latest, balancesheet)
        注：分母<0时，因子值为空
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean().fillna(0)
        income_TTM = self.df_bic["n_income"].rolling(4, min_periods=0).mean().fillna(0)
        total_assets = self.df_bic["total_assets"]
        total_assets[total_assets < 0] = np.nan
        acca_TTM = (act_TTM - income_TTM) / total_assets
        acca = self.drop_duplicated_ann_date(acca_TTM)
        ACCA = self.expand_index(acca, self.df_sb)
        return ACCA

    def NetProfitCashCover(self):
        """
        因子描述：
        净利润现金含量
        计算方法：
        净利润现金含量 = 经营活动产生的现金流量净额(TTM, cashflow) / 归属于母公司所有者的净利润(TTM, income)
        注：分母<0时，因子值为空
        n_income_attr_p
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        attr_p_TTM = self.df_bic["n_income_attr_p"].rolling(4, min_periods=0).mean()
        attr_p_TTM[attr_p_TTM < 0] = np.nan
        act_attr_p_TTM = act_TTM / attr_p_TTM
        net_profit_cash_cover = self.drop_duplicated_ann_date(act_attr_p_TTM)
        NetProfitCashCover = self.expand_index(net_profit_cash_cover, self.df_sb)
        return NetProfitCashCover

    def OperCashInToAsset(self):
        """
        因子描述：
        总资产现金回收率
        计算方法：
        总资产现金回收率 = 经营活动产生的现金流量净额(TTM, cashflow) / 总资产(最近4个季度的平均值)
        注：分母<0时，因子值为空
        """
        act_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean().fillna(0)
        t_assets = self.df_bic['total_assets'].rolling(4, min_periods=0).mean().fillna(0)
        t_assets[t_assets < 0] = np.nan
        OperCashInToAsset_tmp = act_TTM / t_assets
        OperCashInToAsset_tmp = self.drop_duplicated_ann_date(OperCashInToAsset_tmp)
        Oper_CashInToAsset = self.expand_index(OperCashInToAsset_tmp, self.df_sb)
        return Oper_CashInToAsset

    def SalesServiceCashToORLatest(self):
        """
        因子描述：
        销售商品提供劳务收到的现金与营业收入之比（Latest）
        计算方法：
        销售商品提供劳务收到的现金与营业收入之比 = 销售商品提供劳务收到的现金 / 营业收入
        """
        Cfss = self.df_bic["c_fr_sale_sg"].fillna(0)
        Revenue = self.df_bic["revenue"].fillna(0)
        SalesServiceCashToORLatest_tmp = Cfss / Revenue
        SalesServiceCashToORLatest_tmp = self.drop_duplicated_ann_date(SalesServiceCashToORLatest_tmp)
        Sales_ServiceCashToORLatest = self.expand_index(SalesServiceCashToORLatest_tmp, self.df_sb)
        return Sales_ServiceCashToORLatest

    def CashRateOfSalesLatest(self):
        """
        因子描述：
        经营活动产生的现金流量净额与营业收入之比（Latest）
        计算方法：
        经营活动产生的现金流量净额与营业收入之比 = 经营活动产生的现金流量净额 / 营业收入
        """
        act = self.df_bic["n_cashflow_act"].fillna(0)
        Revenue = self.df_bic["revenue"].fillna(0)
        CashRateOfSalesLatest_tmp = act / Revenue
        CashRateOfSalesLatest_tmp = self.drop_duplicated_ann_date(CashRateOfSalesLatest_tmp)
        Cash_RateOfSalesLatest = self.expand_index(CashRateOfSalesLatest_tmp, self.df_sb)
        return Cash_RateOfSalesLatest

    def NOCFToOperatingNILatest(self):
        """
        因子描述：
        经营活动产生的现金流量净额与经营活动净收益之比
        计算方法：
        经营活动产生的现金流量净额与经营活动净收益之比 = 经营活动产生的现金流量净额 / 经营活动净收益
        """
        # act = self.df_bic["n_cashflow_act"].fillna(0)
        # Revenue = self.df_bic["revenue"].fillna(0)
        # CashRateOfSalesLatest_tmp = act /
