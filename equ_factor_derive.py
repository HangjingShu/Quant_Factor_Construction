import logging
import pandas as pd

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_衍生数据 43
class EquFactorDerive(Equ):
    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb

    def TotalFixedAssets(self):
        """
        因子描述：
        固定资产合计。
        计算方法：
        固定资产合计=固定资产+在建工程+工程物质+固定资产清理。
        其中金融企业不计算
        """
        Fix_Assets = self.df_bic["fix_assets"].fillna(0)
        Cip = self.df_bic["cip"].fillna(0)
        Const_Materials = self.df_bic["const_materials"].fillna(0)
        Fixed_Assets_Disp = self.df_bic["fixed_assets_disp"].fillna(0)
        TFA = Fix_Assets + Cip + Const_Materials + Fixed_Assets_Disp
        Total_Fixed_Assets_tmp = self.drop_duplicated_ann_date(TFA)
        Total_Fixed_Assets = self.expand_index(Total_Fixed_Assets_tmp, self.df_sb)
        return Total_Fixed_Assets

    def IntFreeCL(self):
        """
        因子描述：
        无息流动负债。
        计算方法：
        无息流动负债=应付账款+预收账款+应付股利+应付利息+应交税费+应付职工薪酬+其他应付款+预提费用+其他流动负债
        """
        IntFreeCL_tmp = self.df_bic["acct_payable"].fillna(0) + \
                        self.df_bic["adv_receipts"].fillna(0) + \
                        self.df_bic["div_payable"].fillna(0) + \
                        self.df_bic["int_payable"].fillna(0) + \
                        self.df_bic["taxes_payable"].fillna(0) + \
                        self.df_bic["payroll_payable"].fillna(0) + \
                        self.df_bic["oth_payable"].fillna(0) + \
                        self.df_bic["acc_exp"].fillna(0) + \
                        self.df_bic["oth_cur_liab"].fillna(0)
        IntFreeCL_tmp = self.drop_duplicated_ann_date(IntFreeCL_tmp)
        Int_FreeCL = self.expand_index(IntFreeCL_tmp, self.df_sb)
        return Int_FreeCL

    def IntFreeNCL(self):
        """
        因子描述：
        无息非流动负债。
        计算方法：
        无息非流动负债=非流动负债合计-（长期借款+应付债券）
        附注：金融企业不计算
        财务科目的空值用前值填充
        """

        IntFreeNCL_tmp = self.df_bic["total_ncl"].fillna(0) - (
                self.df_bic["lt_borr"].fillna(0) + self.df_bic["bond_payable"].fillna(0))
        IntFreeNCL_tmp = self.drop_duplicated_ann_date(IntFreeNCL_tmp)
        Int_FreeNCL = self.expand_index(IntFreeNCL_tmp, self.df_sb)
        return Int_FreeNCL

    def IntCL(self):
        """
        因子描述：
        带息流动负债
        计算方法：
        带息流动负债=流动负债-无息流动负债
        附注1：金融企业不计算
        附注2：一般工商业公司合并报表中有金融类负债科目看作有息流动负债，例如：吸收存款
        财务科目的空值用前值填充
        """
        IntFreeCL_tmp = self.df_bic["acct_payable"].fillna(0) + \
                        self.df_bic["adv_receipts"].fillna(0) + \
                        self.df_bic["div_payable"].fillna(0) + \
                        self.df_bic["int_payable"].fillna(0) + \
                        self.df_bic["taxes_payable"].fillna(0) + \
                        self.df_bic["payroll_payable"].fillna(0) + \
                        self.df_bic["oth_payable"].fillna(0) + \
                        self.df_bic["acc_exp"].fillna(0) + \
                        self.df_bic["oth_cur_liab"].fillna(0)
        IntCL_tmp = self.df_bic["total_cur_liab"].fillna(0) - IntFreeCL_tmp
        IntCL1 = self.expand_index(self.drop_duplicated_ann_date(IntCL_tmp), self.df_sb)
        return IntCL1

    def IntDebt(self):
        """
        因子描述：
        带息债务
        计算方法：
        带息债务=带息流动负债+长期借款+应付债券
        附注1：金融企业不计算
        附注2：一般工商业公司合并报表中有金融类负债科目看作有息流动负债，例如：吸收存款
        财务科目的空值用前值填充
        """
        IntFreeCL_tmp = self.df_bic["acct_payable"].fillna(0) + \
                        self.df_bic["adv_receipts"].fillna(0) + \
                        self.df_bic["div_payable"].fillna(0) + \
                        self.df_bic["int_payable"].fillna(0) + \
                        self.df_bic["taxes_payable"].fillna(0) + \
                        self.df_bic["payroll_payable"].fillna(0) + \
                        self.df_bic["oth_payable"].fillna(0) + \
                        self.df_bic["acc_exp"].fillna(0) + \
                        self.df_bic["oth_cur_liab"].fillna(0)
        IntCL_tmp = self.df_bic["total_cur_liab"].fillna(0) - IntFreeCL_tmp
        IntDebt_tmp = IntCL_tmp + self.df_bic["lt_borr"].fillna(0) + self.df_bic["bond_payable"].fillna(0) + \
                      self.df_bic["depos"].fillna(0)
        IntDebt1 = self.expand_index(self.drop_duplicated_ann_date(IntDebt_tmp), self.df_sb)
        return IntDebt1

    def NetDebt(self):
        """
        因子描述：
        净债务
        计算方法：
        净债务=带息负债-货币资金
        附注1：金融企业不计算
        附注2：一般工商业公司合并报表中有金融类负债科目看作有息流动负债，例如：吸收存款
        """
        IntFreeCL_tmp = self.df_bic["acct_payable"].fillna(0) + \
                        self.df_bic["adv_receipts"].fillna(0) + \
                        self.df_bic["div_payable"].fillna(0) + \
                        self.df_bic["int_payable"].fillna(0) + \
                        self.df_bic["taxes_payable"].fillna(0) + \
                        self.df_bic["payroll_payable"].fillna(0) + \
                        self.df_bic["oth_payable"].fillna(0) + \
                        self.df_bic["acc_exp"].fillna(0) + \
                        self.df_bic["oth_cur_liab"].fillna(0)
        IntCL_tmp = self.df_bic["total_cur_liab"].fillna(0) - IntFreeCL_tmp
        IntDebt_tmp = IntCL_tmp + self.df_bic["lt_borr"].fillna(0) + self.df_bic["bond_payable"].fillna(0) + \
                      self.df_bic["depos"].fillna(0)
        NetDebt_tmp = IntDebt_tmp - self.df_bic["money_cap"].fillna(0)
        NetDebt1 = self.expand_index(self.drop_duplicated_ann_date(NetDebt_tmp), self.df_sb)
        return NetDebt1

    def NetTangibleAssets(self):
        """
        因子描述：
        有形净资产、有形资产净值
        计算方法：
        有形净资产=归属于母公司的所有者权益-无形资产-研发支出-商誉-长期待摊费用-递延所得税资产
        """
        # 归属于母公司的所有者权益
        NetTangibleAssets_tmp = self.df_bic["total_hldr_eqy_exc_min_int"].fillna(0) - self.df_bic[
            "intan_assets"].fillna(0) - \
                                self.df_bic["r_and_d"].fillna(0) - self.df_bic["goodwill"].fillna(0) - \
                                self.df_bic["lt_amort_deferred_exp"].fillna(0) - self.df_bic["defer_tax_assets"].fillna(
            0)
        NetTangibleAssets_tmp = self.drop_duplicated_ann_date(NetTangibleAssets_tmp)
        NetTangible_Assets = self.expand_index(NetTangibleAssets_tmp, self.df_sb)
        return NetTangible_Assets

    def WorkingCapital(self):
        """
        因子描述：
        运营资本
        计算方法：
        营运资本=流动资产-流动负债
        附注：金融企业不计算
        财务科目的空值用前值填充
        """
        WorkingCapital_tmp = self.df_bic["total_cur_assets"].fillna(0) - self.df_bic["total_cur_liab"].fillna(0)
        WorkingCapital_tmp = self.drop_duplicated_ann_date(WorkingCapital_tmp)
        Working_Capital = self.expand_index(WorkingCapital_tmp, self.df_sb)
        return Working_Capital

    def NetWorkingCapital(self):
        """
        因子描述：
        净运营资本
        计算方法：
        净营运资本=流动资产-货币资金-无息流动负债
        附注1：金融企业不计算
        附注2：一般工商业公司合并报表中有金融类负债科目看作带息流动负债，例如：吸收存款
        财务科目的空值用前值填充
        """
        IntFreeCL_tmp = self.df_bic["acct_payable"].fillna(0) + \
                        self.df_bic["adv_receipts"].fillna(0) + \
                        self.df_bic["div_payable"].fillna(0) + \
                        self.df_bic["int_payable"].fillna(0) + \
                        self.df_bic["taxes_payable"].fillna(0) + \
                        self.df_bic["payroll_payable"].fillna(0) + \
                        self.df_bic["oth_payable"].fillna(0) + \
                        self.df_bic["acc_exp"].fillna(0) + \
                        self.df_bic["oth_cur_liab"].fillna(0)
        NetWorkingCapital_tmp = self.df_bic["total_cur_assets"].fillna(0) - self.df_bic["money_cap"].fillna(0) - \
                                IntFreeCL_tmp
        NetWorkingCapital1 = self.expand_index(self.drop_duplicated_ann_date(NetWorkingCapital_tmp), self.df_sb)
        return NetWorkingCapital1

    def TotalPaidinCapital(self):
        """
        因子描述：
        全部投入资本
        计算方法：
        全部投入资本=所有者权益合计+带息债务
        附注1：金融企业不计算。
        附注2：一般工商业公司合并报表中有金融类负债科目看作带息流动负债，例如：吸收存款。
        财务科目的空值用前值填充
        """
        IntFreeCL_tmp = self.df_bic["acct_payable"].fillna(0) + \
                        self.df_bic["adv_receipts"].fillna(0) + \
                        self.df_bic["div_payable"].fillna(0) + \
                        self.df_bic["int_payable"].fillna(0) + \
                        self.df_bic["taxes_payable"].fillna(0) + \
                        self.df_bic["payroll_payable"].fillna(0) + \
                        self.df_bic["oth_payable"].fillna(0) + \
                        self.df_bic["acc_exp"].fillna(0) + \
                        self.df_bic["oth_cur_liab"].fillna(0)
        IntCL_tmp = self.df_bic["total_cur_liab"].fillna(0) - IntFreeCL_tmp
        IntDebt_tmp = IntCL_tmp + self.df_bic["lt_borr"].fillna(0) + self.df_bic["bond_payable"].fillna(0) + \
                      self.df_bic["depos"].fillna(0)
        Total_Investors_Equity = self.df_bic["total_assets"].fillna(0) - self.df_bic["total_liab"].fillna(0)
        TotalPaidinCapital_tmp = IntDebt_tmp + Total_Investors_Equity
        TotalPaidinCapital1 = self.expand_index(self.drop_duplicated_ann_date(TotalPaidinCapital_tmp), self.df_sb)
        return TotalPaidinCapital1

    def RetainedEarnings(self):
        """
        因子描述：
        留存收益
        计算方法：
        留存收益=盈余公积+未分配利润
        """
        RetainedEarnings_tmp = self.df_bic["surplus_rese"].fillna(0) + self.df_bic["undistr_porfit"].fillna(0)
        RetainedEarnings_tmp = self.drop_duplicated_ann_date(RetainedEarnings_tmp)
        Retained_Earnings = self.expand_index(RetainedEarnings_tmp, self.df_sb)
        return Retained_Earnings

    def OperateNetIncome(self):
        """
        因子描述：
        经营活动净收益
        计算方法：
        对于非金融企业： 经营活动净收益=营业总收入-营业总成本
        对于金融企业： 经营活动净收益=营业收入-公允价值变动损益-投资收益-汇兑损益-营业支出
        """
        pass

    def ValueChgProfit(self):
        """
        因子描述：
        价值变动净收益
        计算方法：
        价值变动净收益=公允价值变动损益+投资收益+汇兑损益
        """

    #     缺少字段

    def NetIntExpense(self):
        """
        因子描述：
        净利息费用
        计算方法：
        净利息费用=利息支出-利息收入（财务费用附注）
        附注1：若未披露财务费用附注，则直接取财务费用值。
        附注2：金融企业不计算
        """
        # NetIntExpense_tmp = self.df_bic["int_exp"].fillna(0) - self.df_bic["int_income"].fillna(0)
        # NetIntExpense_tmp = self.drop_duplicated_ann_date(NetIntExpense_tmp)
        # NetInt_Expense = self.expand_index(NetIntExpense_tmp, self.df_sb)
        # return NetInt_Expense

    def EBIT(self):
        """
        因子描述：
        息税前利润
        计算方法：
        EBIT=利润总额+净利息费用
        附注：金融企业不计算
        """
        EBIT_tmp = self.df_bic["int_exp"].fillna(0) + self.df_bic["int_exp"].fillna(0)
        EBIT_tmp = self.drop_duplicated_ann_date(EBIT_tmp)
        EBIT1 = self.expand_index(EBIT_tmp, self.df_sb)
        return EBIT1

    def EBITDA(self):
        """
        因子描述：
        息税折旧摊销前利润
        计算方法：
        EBITDA=利润总额+净利息费用+固定资产折旧＋无形资产摊销＋长期待摊费用摊销
        附注：金融企业不计算
        """
        EBITDA_tmp = self.df_bic["total_profit"].fillna(0) + \
                     self.df_bic["int_exp"].fillna(0) - self.df_bic["int_income"].fillna(0) + \
                     self.df_bic["depr_fa_coga_dpba"].fillna(0) + \
                     self.df_bic["amort_intang_assets"].fillna(0) + \
                     self.df_bic["lt_amort_deferred_exp"].fillna(0)
        EBITDA_tmp = self.drop_duplicated_ann_date(EBITDA_tmp)
        EBITDA1 = self.expand_index(EBITDA_tmp, self.df_sb)
        return EBITDA1

    def EBIAT(self):
        """
        因子描述：
        息前税后利润
        计算方法：
        EBIAT=利润总额+净利息费用* （1-税率）
        附注1：税率=所得税费用/利润总额（实际税率）； 若所得税费用或利润总额为负，税率=25%（名义税率）
        附注2：金融企业不计算
        """
        tax_rate = self.df_bic["int_exp"].fillna(0) / self.df_bic["total_profit"].fillna(0)
        tax_rate[tax_rate <= 0] = 0.25
        EBIAT_tmp = self.df_bic["total_profit"].fillna(0) + \
                    (self.df_bic["int_exp"].fillna(0) - self.df_bic["int_income"].fillna(0)) * \
                    (1 - tax_rate)
        EBIAT_tmp = self.drop_duplicated_ann_date(EBIAT_tmp)
        EBIAT1 = self.expand_index(EBIAT_tmp, self.df_sb)
        return EBIAT1

    def NRProfitLoss(self):
        """
        因子描述：
        非经常性损益
        计算方法：
        公告中的披露值非经常性损益
        """
        NRProfitLoss_tmp = self.df_bic["non_oper_income"].fillna(0) - self.df_bic["non_oper_exp"].fillna(0)
        NRProfitLoss_tmp = self.drop_duplicated_ann_date(NRProfitLoss_tmp)
        NRProfit_Loss = self.expand_index(NRProfitLoss_tmp, self.df_sb)
        return NRProfit_Loss

    def NIAPCut(self):
        """
        因子描述：
        扣除非经常性损益后的归属于母公司所有者权益净利润
        计算方法：
        直接读取公告中的披露值
        """
        NRProfitLoss_tmp = self.df_bic["non_oper_income"].fillna(0) - self.df_bic["non_oper_exp"].fillna(0)
        NIAPCut_tmp = self.df_bic["n_income_attr_p"] - NRProfitLoss_tmp
        NIAPCut_tmp = self.drop_duplicated_ann_date(NIAPCut_tmp)
        NIAPCut1 = self.expand_index(NIAPCut_tmp, self.df_sb)
        return NIAPCut1

    def FCFF(self):
        """
        因子描述：
        企业自由现金流量。
        计算方法：
        FCFF=经营活动现金流量净额-资本支出-净利息费用* 税率
        附注1：资本支出=购建固定资产、无形资产和其他长期资产支付的现金-处置固定资产、无形资产和其他长期资产收回的
        现金净额。
        附注2：税率=所得税费用/利润总额（实际税率）；若所得税费用或利润总额为负，税率=25%（名义税率）。
        附注3：金融企业不计算

        """
        tax_rate = self.df_bic["int_exp"].fillna(0) / self.df_bic["total_profit"].fillna(0)
        tax_rate[tax_rate <= 0] = 0.25
        Net_interest_expenseself = self.df_bic["int_exp"].fillna(0) - self.df_bic["int_income"].fillna(0)
        Capital_Expenditure = self.df_bic["c_pay_acq_const_fiolta"].fillna(0) - \
                              self.df_bic["n_recp_disp_fiolta"].fillna(0)
        FCFF_tmp = self.df_bic["n_cashflow_act"].fillna(0) - Capital_Expenditure - Net_interest_expenseself * tax_rate
        FCFF_tmp = self.drop_duplicated_ann_date(FCFF_tmp)
        FCFF1 = self.expand_index(FCFF_tmp, self.df_sb)
        return FCFF1

    def FCFE(self):
        """
        因子描述：
        股权自由现金流量
        计算方法：
        FCFE=经营活动现金流量净额-资本支出+债务增加额-净利息费用
        附注1：资本支出=购建固定资产、无形资产和其他长期资产支付的现金-处置固定资产、无形资产和其他长期资产收回的现金净额
        附注2：债务增加额=（期末短期借款+期末长期借款+期末应付债券）-（期初短期借款+期初长期借款+期初应付债券）
        附注3：金融企业不计算
        """
        # Net_interest_expenseself = self.df_bic["int_exp"].fillna(0) - self.df_bic["int_income"].fillna(0)
        # Capital_Expenditure = self.df_bic["c_pay_acq_const_fiolta"].fillna(0) - self.df_bic["n_recp_disp_fiolta"].fillna(0)
        # Increase_debt = self.df_bic["n_cashflow_act"].fillna(0)
        # FCFE_tmp = self.df_bic["n_cashflow_act"].fillna(0) - Capital_Expenditure+

    #     缺少债务增加额

    def DA(self):
        """
        因子描述：
        折旧和摊销
        计算方法：
        折旧及摊销=固定资产折旧＋无形资产摊销＋长期待摊费用摊销
        附注：金融企业不计算
        财务科目的空值用前值填充
        """
        DA_tmp = self.df_bic["depr_fa_coga_dpba"].fillna(0) + \
                 self.df_bic["amort_intang_assets"].fillna(0) + \
                 self.df_bic["lt_amort_deferred_exp"].fillna(0)
        DA_tmp = self.drop_duplicated_ann_date(DA_tmp)
        DA1 = self.expand_index(DA_tmp, self.df_sb)
        return DA1

    def TRevenueTTM(self):
        """
        因子描述：
        营业总收入
        计算方法：
        营业总收入的TTM值
        """
        TRevenueTTM_tmp = self.df_bic["depr_fa_coga_dpba"].rolling(4, min_periods=0).mean()
        TRevenueTTM_tmp = self.drop_duplicated_ann_date(TRevenueTTM_tmp)
        TRevenue_TTM = self.expand_index(TRevenueTTM_tmp, self.df_sb)
        return TRevenue_TTM

    def TCostTTM(self):
        """
        因子描述：
        营业总成本
        计算方法：
        营业总成本的TTM值
        """
        TCostTTM_tmp = self.df_bic["total_cogs"].rolling(4, min_periods=0).mean()
        TCostTTM_tmp = self.drop_duplicated_ann_date(TCostTTM_tmp)
        TCost_TTM = self.expand_index(TCostTTM_tmp, self.df_sb)
        return TCost_TTM

    def RevenueTTM(self):
        """
        因子描述：
        营业收入
        计算方法：
        营业收入的TTM值
        """
        TCostTTM_tmp = self.df_bic["total_cogs"].rolling(4, min_periods=0).mean()
        TCostTTM_tmp = self.drop_duplicated_ann_date(TCostTTM_tmp)
        TCost_TTM = self.expand_index(TCostTTM_tmp, self.df_sb)
        return TCost_TTM

    def CostTTM(self):
        """
        因子描述：
        营业成本
        计算方法：
        营业成本的TTM值
        """
        CostTTM_tmp = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        CostTTM_tmp = self.drop_duplicated_ann_date(CostTTM_tmp)
        Cost_TTM = self.expand_index(CostTTM_tmp, self.df_sb)
        return Cost_TTM

    def GrossProfitTTM(self):
        """
        因子描述：
        毛利
        计算方法：
        毛利的TTM值
        """
        GrossProfit_tmp = self.df_bic["revenue"].fillna(0) - self.df_bic["oper_cost"].fillna(0)
        GrossProfitTTM_tmp = GrossProfit_tmp.rolling(4, min_periods=0).mean()
        GrossProfitTTM_tmp = self.drop_duplicated_ann_date(GrossProfitTTM_tmp)
        GrossProfit_TTM = self.expand_index(GrossProfitTTM_tmp, self.df_sb)
        return GrossProfit_TTM

    def SalesExpenseTTM(self):
        """
        因子描述：
        销售费用
        计算方法：
        销售费用的TTM值
        """
        SalesExpenseTTM_tmp = self.df_bic["sell_exp"].rolling(4, min_periods=0).mean()
        SalesExpenseTTM_tmp = self.drop_duplicated_ann_date(SalesExpenseTTM_tmp)
        SalesExpense_TTM = self.expand_index(SalesExpenseTTM_tmp, self.df_sb)
        return SalesExpense_TTM

    def AdminExpenseTTM(self):
        """
        因子描述：
        管理费用
        计算方法：
        管理费用的TTM值
        """
        AdminExpenseTTM_tmp = self.df_bic["admin_exp"].rolling(4, min_periods=0).mean()
        AdminExpenseTTM_tmp = self.drop_duplicated_ann_date(AdminExpenseTTM_tmp)
        AdminExpense_TTM = self.expand_index(AdminExpenseTTM_tmp, self.df_sb)
        return AdminExpense_TTM

    def FinanExpenseTTM(self):
        """
        因子描述：
        财务费用
        计算方法：
        财务费用的TTM值
        """
        FinanExpenseTTM_tmp = self.df_bic["fin_exp"].rolling(4, min_periods=0).mean()
        FinanExpenseTTM_tmp = self.drop_duplicated_ann_date(FinanExpenseTTM_tmp)
        FinanExpense_TTM = self.expand_index(FinanExpenseTTM_tmp, self.df_sb)
        return FinanExpense_TTM

    def AssetImpairLossTTM(self):
        """
        因子描述：
        资产减值损失
        计算方法：
        资产减值损失的TTM值
        """
        AssetImpairLossTTM_tmp = self.df_bic["assets_impair_loss"].rolling(4, min_periods=0).mean()
        AssetImpairLossTTM_tmp = self.drop_duplicated_ann_date(AssetImpairLossTTM_tmp)
        AssetImpairLoss_TTM = self.expand_index(AssetImpairLossTTM_tmp, self.df_sb)
        return AssetImpairLoss_TTM

    def NPFromOperatingTTM(self):
        """
        因子描述：
        经营活动净收益
        计算方法：
        经营活动净收益的TTM值
        """
        # NPFromOperatingTTM_tmp = self.df_bic["total_revenue"] - self.df_bic["total_cogs"]
        # NPFromOperatingTTM_tmp = NPFromOperatingTTM_tmp.rolling(4, min_periods=0).mean()
        # NPFromOperatingTTM_tmp = self.drop_duplicated_ann_date(NPFromOperatingTTM_tmp)
        # NPFromOperating_TTM = self.expand_index(NPFromOperatingTTM_tmp, self.df_sb)
        # return NPFromOperating_TTM

    # 无经营活动净收益

    def NPFromValueChgTTM(self):
        """
        因子描述：
        价值变动净收益
        计算方法：
        价值变动净收益的TTM值
        """
        NPFromValueChgTTM_tmp = self.df_bic["fv_value_chg_gain"].rolling(4, min_periods=0).mean()
        NPFromValueChgTTM_tmp = self.drop_duplicated_ann_date(NPFromValueChgTTM_tmp)
        NPFromValueChg_TTM = self.expand_index(NPFromValueChgTTM_tmp, self.df_sb)
        return NPFromValueChg_TTM

    def OperateProfitTTM(self):
        """
        因子描述：
        营业利润
        计算方法：
        营业利润的TTM值
        """
        OperateProfitTTM_tmp = self.df_bic["operate_profit"].rolling(4, min_periods=0).mean()
        OperateProfitTTM_tmp = self.drop_duplicated_ann_date(OperateProfitTTM_tmp)
        OperateProfit_TTM = self.expand_index(OperateProfitTTM_tmp, self.df_sb)
        return OperateProfit_TTM

    def NonOperatingNPTTM(self):
        """
        因子描述：
        营业外收支净额
        计算方法：
        营业外收支净额 = 营业外收入TTM-营业外支出TTM
        """
        income = self.df_bic["non_oper_income"].rolling(4, min_periods=0).mean()
        exp = self.df_bic["non_oper_exp"].rolling(4, min_periods=0).mean()
        NonOperatingNPTTM_tmp = income - exp
        NonOperatingNPTTM_tmp = self.drop_duplicated_ann_date(NonOperatingNPTTM_tmp)
        NonOperatingNP_TTM = self.expand_index(NonOperatingNPTTM_tmp, self.df_sb)
        return NonOperatingNP_TTM

    def TProfitTTM(self):
        """
        因子描述：
        利润总额
        计算方法：
        利润总额的TTM值
        """
        TProfitTTM_tmp = self.df_bic["total_profit"].rolling(4, min_periods=0).mean()
        TProfitTTM_tmp = self.drop_duplicated_ann_date(TProfitTTM_tmp)
        TProfit_TTM = self.expand_index(TProfitTTM_tmp, self.df_sb)
        return TProfit_TTM

    def NetProfitTTM(self):
        """
        因子描述：
        净利润
        计算方法：
        净利润的TTM值
        """
        NetProfitTTM_tmp = self.df_bic["n_income"].rolling(4, min_periods=0).mean()
        NetProfitTTM_tmp = self.drop_duplicated_ann_date(NetProfitTTM_tmp)
        NetProfit_TTM = self.expand_index(NetProfitTTM_tmp, self.df_sb)
        return NetProfit_TTM

    def NetProfitAPTTM(self):
        """
        因子描述：
        归属于母公司股东的净利润
        计算方法：
        归属于母公司股东的净利润的TTM值
        """
        NetProfitAPTTM_tmp = self.df_bic["n_income_attr_p"].rolling(4, min_periods=0).mean()
        NetProfitAPTTM_tmp = self.drop_duplicated_ann_date(NetProfitAPTTM_tmp)
        NetProfitAP_TTM = self.expand_index(NetProfitAPTTM_tmp, self.df_sb)
        return NetProfitAP_TTM

    def SaleServiceRenderCashTTM(self):
        """
        因子描述：
        销售商品提供劳务收到的现金
        计算方法：
        销售商品提供劳务收到的现金的TTM值
        """
        SaleServiceRenderCashTTM_tmp = self.df_bic["c_fr_sale_sg"].rolling(4, min_periods=0).mean()
        SaleServiceRenderCashTTM_tmp = self.drop_duplicated_ann_date(SaleServiceRenderCashTTM_tmp)
        SaleServiceRenderCash_TTM = self.expand_index(SaleServiceRenderCashTTM_tmp, self.df_sb)
        return SaleServiceRenderCash_TTM

    def NetOperateCFTTM(self):
        """
        因子描述：
        经营活动现金流量净额
        计算方法：
        经营活动现金流量净额的TTM值
        """
        NetOperateCF_TTM = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean()
        NetOperateCF_TTM1 = self.expand_index(NetOperateCF_TTM, self.df_sb)
        return NetOperateCF_TTM1

    def NetInvestCFTTM(self):
        """
        因子描述：
        投资活动现金流量净额
        计算方法：
        投资活动现金流量净额的TTM值
        """
        NetInvestCFTTM_tmp = self.df_bic["n_cashflow_inv_act"].rolling(4, min_periods=0).mean()
        NetInvestCFTTM_tmp = self.drop_duplicated_ann_date(NetInvestCFTTM_tmp)
        NetInvestCF_TTM = self.expand_index(NetInvestCFTTM_tmp, self.df_sb)
        return NetInvestCF_TTM

    def NetFinanceCFTTM(self):
        """
        因子描述：
        筹资活动现金流量净额
        计算方法：
        筹资活动现金流量净额的TTM值
        """
        NetFinanceCFTTM_tmp = self.df_bic["n_cash_flows_fnc_act"].rolling(4, min_periods=0).mean()
        NetFinanceCFTTM_tmp = self.drop_duplicated_ann_date(NetFinanceCFTTM_tmp)
        NetFinanceCF_TTM = self.expand_index(NetFinanceCFTTM_tmp, self.df_sb)
        return NetFinanceCF_TTM

    def GrossProfit(self):
        """
        因子描述：
        毛利
        计算方法：
        毛利=营业收入-营业成本
        附注：金融企业不计算
        """
        GrossProfit_tmp = self.df_bic["revenue"].fillna(0) - self.df_bic["oper_cost"].fillna(0)
        GrossProfit_tmp = self.drop_duplicated_ann_date(GrossProfit_tmp)
        Gross_Profit = self.expand_index(GrossProfit_tmp, self.df_sb)
        return Gross_Profit

