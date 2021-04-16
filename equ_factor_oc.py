import pandas as pd

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_营运能力 12
class EquFactorOc(Equ):
    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb

    def InventoryTRate(self):
        """
        因子描述：
        存货周转率（Inventory turnover ratio）。
        计算方法：
        存货周转率=营业成本/存货

        符号说明：
        符号 描述 计算方法
        OC 营业成本 TTM
        I 存货 取最近4次记录的平均值
        """
        Oper_Cost = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        Inventories = self.df_bic["inventories"].rolling(4, min_periods=0).mean()
        Inventory_T_Rate_tmp = Oper_Cost/Inventories
        Inventory_T_Rate_tmp = self.drop_duplicated_ann_date(Inventory_T_Rate_tmp)
        Inventory_T_Rate = self.expand_index(Inventory_T_Rate_tmp, self.df_sb)
        return Inventory_T_Rate


    def InventoryTDays(self):
        """
        因子描述：
        存货周转天数（Inventory turnover days）。
        计算方法：
        存货周转天数=360/存货周转率
        """
        Oper_Cost = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        Inventories = self.df_bic["inventories"].rolling(4, min_periods=0).mean()
        Inventory_T_Days_tmp = 360/(Oper_Cost/Inventories)
        Inventory_T_Days_tmp = self.drop_duplicated_ann_date(Inventory_T_Days_tmp)
        Inventory_T_Days = self.expand_index(Inventory_T_Days_tmp, self.df_sb)
        return Inventory_T_Days


    def ARTRate(self):
        """
        因子描述：
        应收账款周转率（Accounts receivable turnover ratio）。
        计算方法：
        符号说明：
        符号 描述 计算方法
        OR 营业收入 TTM
        AccR 应收账款 LATEST(mean)
        BillR 应收票据 LATEST(mean)
        AdvR 预收账款 LATEST(mean)
        空值处理方法同FixedAssetsTRate
        """
        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        AccR = self.df_bic["accounts_receiv"].rolling(4, min_periods=1).mean().fillna(0)
        BillR = self.df_bic["notes_receiv"].rolling(4, min_periods=1).mean().fillna(0)
        AdvR = self.df_bic["adv_receipts"].rolling(4, min_periods=1).mean().fillna(0)
        ART_Rate_tmp = OR/(AccR+BillR-AdvR)
        ART_Rate_tmp = self.drop_duplicated_ann_date(ART_Rate_tmp)
        ART_Rate = self.expand_index(ART_Rate_tmp, self.df_sb)
        return ART_Rate

    def ARTDays(self):
        """
        因子描述：
        应收账款周转天数（Accounts receivable turnover days ）。
        计算方法：
        应收账款周转天数=360/应收账款周转率
        """
        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        AccR = self.df_bic["accounts_receiv"].rolling(4, min_periods=1).mean().fillna(0)
        BillR = self.df_bic["notes_receiv"].rolling(4, min_periods=1).mean().fillna(0)
        AdvR = self.df_bic["adv_receipts"].rolling(4, min_periods=1).mean().fillna(0)
        ART_Days_tmp = 360/(OR/(AccR+BillR-AdvR))
        ART_Days_tmp = self.drop_duplicated_ann_date(ART_Days_tmp)
        ART_Days = self.expand_index(ART_Days_tmp, self.df_sb)
        return ART_Days

    def AccountsPayablesTRate(self):
        """
        因子描述：
        应付账款周转率（Accounts payable turnover rate）。
        计算方法：
        符号说明：
        符号 描述 计算方法
        OC 营业成本 TTM
        AccP 应付账款 取最近4次记录的平均值
        NoteP 应付票据 取最近4次记录的平均值
        AdvP 预付款项 取最近4次记录的平均值
        """
        Oper_Cost = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        AccP = self.df_bic["acct_payable"].rolling(4).mean().fillna(0)
        NoteP = self.df_bic["notes_payable"].rolling(4).mean().fillna(0)
        AdvP = self.df_bic["prepayment"].rolling(4).mean().fillna(0)
        Accounts_Payables_TRate_tmp = Oper_Cost / (AccP+NoteP+AdvP)
        Accounts_Payables_TRate_tmp = self.drop_duplicated_ann_date(Accounts_Payables_TRate_tmp)
        Accounts_Payables_TRate = self.expand_index(Accounts_Payables_TRate_tmp, self.df_sb)
        return Accounts_Payables_TRate

    def AccountsPayablesTDays(self):
        """
        因子描述：
        应付账款周转天数（Accounts payable turnover days）。
        计算方法：
        应付账款周转天数=360/应付账款周转率
        """
        Oper_Cost = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        AccP = self.df_bic["acct_payable"].rolling(4, min_periods=0).mean().fillna(0)
        NoteP = self.df_bic["notes_payable"].rolling(4, min_periods=0).mean().fillna(0)
        AdvP = self.df_bic["prepayment"].rolling(4, min_periods=0).mean().fillna(0)
        AccountsPayablesTDays_tmp = 360 / (Oper_Cost / (AccP+NoteP+AdvP))
        AccountsPayablesTDays_tmp = self.drop_duplicated_ann_date(AccountsPayablesTDays_tmp)
        AccountsPayablesT_Days = self.expand_index(AccountsPayablesTDays_tmp, self.df_sb)
        return AccountsPayablesT_Days

    def CurrentAssetsTRate(self):
        """
        因子描述：
        流动资产周转率（Current assets turnover ratio）
        计算方法：
        符号说明：
        符号 描述 计算方法
        OR 营业收入 TTM
        TCA 流动资产合计 取最近4次记录的平均值
        """

        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        TCA = self.df_bic["total_cur_assets"].rolling(window=4).mean()
        CurrentAssetsTRate_tmp = OR / TCA
        CurrentAssetsTRate_tmp = self.drop_duplicated_ann_date(CurrentAssetsTRate_tmp)
        CurrentAssets_TRate = self.expand_index(CurrentAssetsTRate_tmp, self.df_sb)
        return CurrentAssets_TRate

    def FixedAssetsTRate(self):
        """
        因子描述：
        固定资产周转率（Fixed assets turnover ratio ）。
        计算方法：
        符号说明：
        符号 描述 计算方法
        OR 营业收入 TTM
        FA 固定资产 LATEST
        CM 工程物资 LATEST
        CIP 在建工程 LATEST
        符号 描述 计算方法
        分母的三个会计科目同时为空时，取上一个有正常值披露的财务期的值（只要有一个非空就符合条件，其余的科目空值用
        0填充，如最新会计期为2018Q2，披露值都为空，之前的披露都至少有一个会计科目有值，则取
        2018Q1，2017Q4，2017Q3，2017Q2 4个会计期的值进行计算，空值用0填充）
        """

        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        FA = self.df_bic["fix_assets"].rolling(4, min_periods=1).mean().fillna(0)
        CM = self.df_bic["const_materials"].rolling(4, min_periods=1).mean().fillna(0)
        CIP = self.df_bic["cip"].rolling(4, min_periods=1).mean().fillna(0)
        FixedAssetsTRate_tmp = OR/(FA+CM+CIP)
        FixedAssetsTRate_tmp = self.drop_duplicated_ann_date(FixedAssetsTRate_tmp)
        FixedAssets_TRate = self.expand_index(FixedAssetsTRate_tmp, self.df_sb)
        return FixedAssets_TRate

    def EquityTRate(self):
        """
        因子描述：
        股东权益周转率（Equity turnover ratio）。
        计算方法：
        符号说明：
        符号 描述 计算方法
        OR 营业收入 TTM
        TSE 股东权益 取最近4次记录的平均值(有问题)？？？？
        """
        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        TSE = self.df_bic["total_hldr_eqy_inc_min_int"].rolling(4, min_periods=1).mean().fillna(0)
        EquityTRate_tmp = OR/TSE
        EquityTRate_tmp = self.drop_duplicated_ann_date(EquityTRate_tmp)
        EquityT_Rate = self.expand_index(EquityTRate_tmp, self.df_sb)
        return EquityT_Rate

    def TotalAssetsTRate(self):
        """
        因子描述：
        总资产周转率（Total assets turnover ratio）。
        计算方法：
        符号说明：
        符号 描述 计算方法
        OR 营业收入 TTM
        TA 总资产 取最近4个季度的平均值
        """
        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        TCA = self.df_bic["total_cur_assets"].rolling(4, min_periods=1).mean().fillna(0)
        TNCA = self.df_bic["total_nca"].rolling(4, min_periods=1).mean().fillna(0)
        TotalAssetsTRate_tmp = OR/(TCA + TNCA)
        TotalAssetsTRate_tmp = self.drop_duplicated_ann_date(TotalAssetsTRate_tmp)
        TotalAssets_TRate = self.expand_index(TotalAssetsTRate_tmp, self.df_sb)
        return TotalAssets_TRate

    def CashConversionCycle(self):
        """
        因子描述：
        现金转换周期。
        计算方法：
        现金转换周期=应收账款周转天数+存货周转天数-应付账款周转天数
        """
        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        AccR = self.df_bic["accounts_receiv"].rolling(4, min_periods=0).mean().fillna(0)
        BillR = self.df_bic["notes_receiv"].rolling(4, min_periods=0).mean().fillna(0)
        AdvR = self.df_bic["adv_receipts"].rolling(4, min_periods=0).mean().fillna(0)
        ART_Days_tmp = 360/(OR/(AccR+BillR-AdvR))
        Oper_Cost = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        Inventories = self.df_bic["inventories"].rolling(4, min_periods=0).mean()
        Inventory_T_Days_tmp = 360/(Oper_Cost/Inventories)
        Oper_Cost = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        AccP = self.df_bic["acct_payable"].rolling(4, min_periods=0).mean().fillna(0)
        NoteP = self.df_bic["notes_payable"].rolling(4, min_periods=0).mean().fillna(0)
        AdvP = self.df_bic["prepayment"].rolling(4, min_periods=0).mean().fillna(0)
        AccountsPayablesTDays_tmp = 360 / (Oper_Cost / (AccP + NoteP + AdvP))
        CashConversionCycle_tmp = ART_Days_tmp + Inventory_T_Days_tmp - AccountsPayablesTDays_tmp
        CashConversionCycle_tmp = self.drop_duplicated_ann_date(CashConversionCycle_tmp)
        CashConversion_Cycle = self.expand_index(CashConversionCycle_tmp, self.df_sb)
        return CashConversion_Cycle

    def OperatingCycle(self):
        """
        因子描述：
        营业周期。
        计算方法：
        营业周期 = 应收账款周转天数+存货周转天数
        """
        OR = self.df_bic["revenue"].rolling(4, min_periods=0).mean()
        AccR = self.df_bic["accounts_receiv"].rolling(4, min_periods=1).mean().fillna(0)
        BillR = self.df_bic["notes_receiv"].rolling(4, min_periods=1).mean().fillna(0)
        AdvR = self.df_bic["adv_receipts"].rolling(4, min_periods=1).mean().fillna(0)
        ART_Days_tmp = 360/(OR/(AccR+BillR-AdvR))
        Oper_Cost = self.df_bic["oper_cost"].rolling(4, min_periods=0).mean()
        Inventories = self.df_bic["inventories"].rolling(4, min_periods=0).mean()
        Inventory_T_Days_tmp = 360/(Oper_Cost/Inventories)
        OperatingCycle_tmp = ART_Days_tmp + Inventory_T_Days_tmp
        OperatingCycle_tmp = self.drop_duplicated_ann_date(OperatingCycle_tmp)
        Operating_Cycle = self.expand_index(OperatingCycle_tmp, self.df_sb)
        return Operating_Cycle

    # def UPDATE_TIME(self):
    #     pass

