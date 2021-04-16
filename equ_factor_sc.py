import pandas as pd

"""
get series from df_bic
calculate if necessary
drop duplicated records
expand the index list of df_bic by df_sb
div df_sb if necessary
"""


# 股票因子_偿债能力和资本结构 27
class EquFactorSc(Equ):
    def __init__(self, bic, sb):
        super().__init__()
        self.bic = bic
        self.sb = sb

    def CurrentRatio(self):
        """
        因子描述
        流动比率（Current ratio），
        计算方法：流动比率=流动资产合计/流动负债合计
        """
        total_CurrentAsset = self.df_bic["total_cur_assets"].fillna(0)
        total_CurrentLiability = self.df_bic["total_cur_liab"].fillna(0)
        Current_Ratio = total_CurrentAsset / total_CurrentLiability
        CurrentRatio_D = self.drop_duplicated_ann_date(Current_Ratio)
        CurrentRatio = self.expand_index(CurrentRatio_D, self.df_sb)
        return CurrentRatio

    def QuickRatio(self):
        """
        因子描述：
        速动比率（Quick ratio）。
        计算方法：
        速动比率=(流动资产合计-存货)/ 流动负债合计
        """
        total_CurrentAsset = self.df_bic["total_cur_assets"].fillna(0)
        Inventory = self.df_bic["inventories"].fillna(0)
        total_CurrentLiability = self.df_bic["total_cur_liab"].fillna(0)
        QuickRatio = (total_CurrentAsset - Inventory) / total_CurrentLiability
        Quickratio = self.drop_duplicated_ann_date(QuickRatio)
        Quick_Ratio = self.expand_index(Quickratio, self.df_sb)
        return Quick_Ratio

    def DebtEquityRatio(self):
        """
        因子描述：
        产权比率（Debt equity ratio）。
        计算方法：
        产权比率 = 负债合计 / 归属母公司所有者权益合计 (股东权益合计(不含少数股东权益))
        """
        total_liability = self.df_bic["total_liab"].fillna(0)
        Equity = self.df_bic["total_hldr_eqy_exc_min_int"].fillna(0)
        debtequityratio = total_liability / Equity
        DER = self.drop_duplicated_ann_date(debtequityratio)
        DebtEquityRatio = self.expand_index(DER, self.df_sb)
        return DebtEquityRatio

    def LongDebtToWorkingCapital(self):
        """
        因子描述：
        长期负债与营运资金比率（Long term debt to working capital）。
        计算方法：
        长期负债与营运资金比率=非流动负债合计/(流动资产合计-流动负债合计)
        """
        total_NonCurrentLiability = self.df_bic["total_ncl"].fillna(0)
        total_CurrentAsset = self.df_bic["total_cur_assets"].fillna(0)
        total_CurrentLiability = self.df_bic["total_cur_liab"].fillna(0)
        LTD2WC = total_NonCurrentLiability / (total_CurrentAsset - total_CurrentLiability)
        L = self.drop_duplicated_ann_date(LTD2WC)
        LTD_WC = self.expand_index(L, self.df_sb)
        return LTD_WC

    def EquityFixedAssetRatio(self):
        """
        因子描述：
        股东权益与固定资产比率（Equity fixed assets ratio）。
        计算方法：
        股东权益与固定资产比率=股东权益/(固定资产+工程物资+在建工程)
        """
        EL = self.df_bic[["total_hldr_eqy_inc_min_int", "fix_assets", "const_materials", "cip"]].copy()
        EL = EL.fillna(method="pad")
        EquityFixedAsset_Ratio = EL["total_hldr_eqy_inc_min_int"] / (
                EL["fix_assets"] + EL["const_materials"] + EL["cip"])
        EFAR = self.drop_duplicated_ann_date(EquityFixedAsset_Ratio)
        EquityFixedAssetRatio = self.expand_index(EFAR, self.df_sb)
        return EquityFixedAssetRatio

    def LongDebtToAsset(self):
        """
        因子描述：
        长期借款与资产总计之比（Long term loan to total assets）。
        计算方法：
        长期借款与资产总计之比=长期借款/总资产
        """
        LongTermBorrow = self.df_bic["lt_borr"].fillna(0)
        total_asset = self.df_bic["total_assets"].fillna(0)
        LTL2TA = LongTermBorrow / total_asset
        LongDebtTo_Asset = self.drop_duplicated_ann_date(LTL2TA)
        LongDebtToAsset = self.expand_index(LongDebtTo_Asset, self.df_sb)
        return LongDebtToAsset

    def BondsPayableToAsset(self):
        """
        因子描述：
        应付债券与总资产之比（Bonds payable to total assets）。
        计算方法：
        应付债券与总资产之比=应付债券/总资产
        注：财务科目的空值用前值填充
        """
        LA = self.df_bic[["bond_payable", "total_assets"]].copy()
        LA = LA.fillna(method="pad")
        BondsPayableToAsset = LA["bond_payable"] / LA["total_assets"]
        BPTA = self.drop_duplicated_ann_date(BondsPayableToAsset)
        BondsPayabletoAsset = self.expand_index(BPTA, self.df_sb)
        return BondsPayabletoAsset

    def LongTermDebtToAsset(self):
        """
        因子描述：
        长期负债与资产总计之比（Long term debt to total assets）。
        计算方法：
        长期负债与资产总计之比=非流动负债合计/总资产
        """
        total_nonCurrentLiability = self.df_bic["total_ncl"].fillna(0)
        total_asset = self.df_bic["total_assets"].fillna(0)
        LongTermDebtToasset = total_nonCurrentLiability / total_asset
        LTD2A = self.drop_duplicated_ann_date(LongTermDebtToasset)
        LongTermDebtToAsset = self.expand_index(LTD2A, self.df_sb)
        return LongTermDebtToAsset

    def EquityToAsset(self):
        """
        因子描述：
        股东权益比率（Equity to total assets）。
        计算方法：
        股东权益比率=股东权益/总资产
        """
        Equity = self.df_bic["total_hldr_eqy_inc_min_int"].fillna(0)
        total_Asset = self.df_bic["total_assets"].fillna(0)
        EquitytoAsset = Equity / total_Asset
        E2A = self.drop_duplicated_ann_date(EquitytoAsset)
        EquityToAsset = self.expand_index(E2A, self.df_sb)
        return EquityToAsset

    def CurrentAssetsRatio(self):
        """
        因子描述：
        流动资产比率（Current assets ratio）。
        计算方法：
        流动资产比率=流动资产合计/总资产
        """
        total_CurrentAsset = self.df_bic["total_cur_assets"].fillna(0)
        total_asset = self.df_bic["total_assets"].fillna(0)
        CurrentAssets_Ratio = total_CurrentAsset / total_asset
        CAR = self.drop_duplicated_ann_date(CurrentAssets_Ratio)
        CurrentAssetsRatio = self.expand_index(CAR, self.df_sb)
        return CurrentAssetsRatio

    def NonCurrentAssetsRatio(self):
        """
        因子描述：
        非流动资产比率（Non-current assets ratio）。
        计算方法：
        非流动资产比率=非流动资产合计/总资产
        """
        total_nonCurrentAsset = self.df_bic["total_nca"].fillna(0)
        total_asset = self.df_bic["total_assets"].fillna(0)
        nonCurrentAssetsRatio = total_nonCurrentAsset / total_asset
        NCAR = self.drop_duplicated_ann_date(nonCurrentAssetsRatio)
        NonCurrentAssetsRatio = self.expand_index(NCAR, self.df_sb)
        return NonCurrentAssetsRatio

    def FixAssetRatio(self):
        """
        因子描述：
        固定资产比率（Fixed assets ratio）。
        计算方法：
        固定资产比率=(固定资产+工程物资+在建工程)/总资产
        注；财务科目的空值用前值填充
        """
        Asset = self.df_bic[["fix_assets", "const_materials", "cip", "total_assets"]].copy()
        Asset = Asset.fillna(axis=0, method="ffill")
        fixAssetRatio = (Asset["fix_assets"] + Asset["const_materials"] + Asset["cip"]) / Asset["total_assets"]
        FAR = self.drop_duplicated_ann_date(fixAssetRatio)
        FixAssetRatio = self.expand_index(FAR, self.df_sb)
        return FixAssetRatio

    def IntangibleAssetRatio(self):
        """
        因子描述：
        无形资产比率（Intangible assets ratio）。
        计算方法：
        无形资产比率=(无形资产+研发支出+商誉)/总资产
        注：财务科目用0填充空值，如果分子三项都是空值，则取前一个有披露值的财报期的值
        """
        Asset = self.df_bic[["intan_assets", "r_and_d", "goodwill", "total_assets"]].copy()
        Signal = (Asset["intan_assets"].isnull()) & (Asset["r_and_d"].isnull()) & (Asset["goodwill"].isnull())
        Asset[Signal == False] = Asset[Signal == False].fillna(0)
        Asset = Asset.fillna(axis=0, method="ffill")
        intangibleAssetRatio = (Asset["intan_assets"] + Asset["r_and_d"] + Asset["goodwill"]) / Asset["total_assets"]
        IAR = self.drop_duplicated_ann_date(intangibleAssetRatio)
        IntangibleAssetRatio = self.expand_index(IAR, self.df_sb)
        return IntangibleAssetRatio

    def BLEV(self):
        """
        因子描述：
        账面杠杆（Book leverage）。
        计算方法：
        账面杠杆=非流动负债合计/(归属于母公司所有者权益合计-其他权益合计)
        “其他权益工具”包括“优先股”，“永续债”等不归属于普通股股东的所有者权益。
        """
        NCL = self.df_bic["total_ncl"].fillna(0)
        total_EquityAttrParent = self.df_bic["total_hldr_eqy_exc_min_int"].fillna(0)
        other_Equity = self.df_bic["oth_eqt_tools"].fillna(0)
        BL = NCL / (total_EquityAttrParent - other_Equity)
        BookLeverage = self.drop_duplicated_ann_date(BL)
        BLEV = self.expand_index(BookLeverage, self.df_bic)
        return BLEV

    def DebtsAssetRatio(self):
        """
        因子描述：
        债务总资产比（Debt to total assets）。
        计算方法：
        债务总资产比=负债合计/总资产
        """
        total_liability = self.df_bic["total_liab"].fillna(0)
        total_asset = self.df_bic["total_assets"].fillna(0)
        debtsAssetRatio = total_liability / total_asset
        DER = self.drop_duplicated_ann_date(debtsAssetRatio)
        DebtsAssetRatio = self.expand_index(DER, self.df_sb)
        return DebtsAssetRatio

    def MLEV(self):
        """
        因子描述：
        市场杠杆（Market leverage）。
        计算方法：
        市场杠杆=(非流动负债合计+总市值+其他权益合计)/总市值
        注：“其他权益工具”包括“优先股”，“永续债”等不归属于普通股股东的所有者权益。
        """
        total_nonCurrentLiability = self.df_bic["total_ncl"].fillna(0)
        tncl = self.drop_duplicated_ann_date(total_nonCurrentLiability)
        TNCL = self.expand_index(tncl, self.df_bic)
        other_Equity = self.df_bic["oth_eqt_tools"].fillna(0)
        oe = self.drop_duplicated_ann_date(other_Equity)
        OE = self.expand_index(oe, self.df_bic)
        total_MV = self.df_sb["total_mv"].fillna(0)
        MLEV = (TNCL + total_MV + OE) / total_MV
        return MLEV

    def InterestCover(self):
        """
        因子描述：
        利息保障倍数。
        计算方法：
        利息保障倍数 = 息税前利润（EBIT）/利息费用
        其中:
        息税前利润=利润总额TTM+利息费用TTM;
        利息费用=利息支出TTM-利息收入TTM（利息费用批注只有半年报、年报有，一季报和三季报用财务费用代替），金融
        类企业不计算该指标（金融类企业指申万一级行业分类中的银行和非银金融）
        注：上面的利息支出和利息收入均取自利润表附注-财务费用
        """

    def SuperQuickRatio(self):
        """
        因子描述：
        超速动比率。
        计算方法：
        超速动比率=（货币资金+交易性金融资产+应收票据+应收帐款+其他应收款）／流动负债合计
        """
        money_cap = self.df_bic["money_cap"].fillna(0)
        TradingAsset = self.df_bic["trad_asset"].fillna(0)
        Notes_receive = self.df_bic["notes_receiv"].fillna(0)
        Accounts_receive = self.df_bic["accounts_receiv"].fillna(0)
        Other_receive = self.df_bic["oth_receiv"].fillna(0)
        total_CurrentLiability = self.df_bic["total_cur_liab"].fillna(0)
        superQuickRatio = (
                                  money_cap + TradingAsset + Notes_receive + Accounts_receive + Other_receive) / total_CurrentLiability
        SQR = self.drop_duplicated_ann_date(superQuickRatio)
        SuperQuickRatio = self.expand_index(SQR, self.df_sb)
        return SuperQuickRatio

    def TSEPToInterestBearDebt(self):
        """
        因子描述：
        归属母公司股东的权益/带息负债。
        计算方法：
        TSEPToInterestBearDebt=归属母公司股东的权益/带息负债
        注：财务科目的空值用前值填充
        带息负债 = 短期借款+一年内到期的长期负债+长期借款+应付债券+（其他应付款）
        """
        St_B = self.df_bic["st_borr"].fillna(method="pad")
        non_Cur_l = self.df_bic["non_cur_liab_due_1y"].fillna(method="pad")
        LT_debt = self.df_bic['lt_borr'].fillna(method="pad")
        Bond_payable = self.df_bic["bond_payable"].fillna(method="pad")
        other_payable = self.df_bic["oth_payable"].fillna(method="pad")
        InterestBearDebt = St_B + non_Cur_l + LT_debt + Bond_payable + other_payable
        EqytoP = self.df_bic["total_hldr_eqy_exc_min_int"].fillna(method="pad")
        TSEPtoInterestBearDebt = EqytoP / InterestBearDebt
        TSEP = self.drop_duplicated_ann_date(TSEPtoInterestBearDebt)
        TSEPToInterestBearDebt = self.expand_index(TSEP, self.df_sb)
        return TSEPToInterestBearDebt

    def DebtTangibleEquityRatio(self):
        """
        因子描述：
        负债合计/有形净值，有形净值债务率。
        计算方法：
        DebtTangibleEquityRatio=负债合计(TL)/有形净值(NTanAssets) 计算方法均为latest
        有形净值 = 固定资产+在建工程
        注：财务科目的空值用前值填充
        """
        df = self.df_bic[['total_liab', 'fix_assets', 'cip']].copy().fillna(method="pad").fillna(0)
        NTanAssets = df['fix_assets'] + df['cip']
        debtTangibleEquityRatio = df['total_liab'] / NTanAssets
        DTER = self.drop_duplicated_ann_date(debtTangibleEquityRatio)
        DebtTangibleEquityRatio = self.expand_index(DTER, self.df_sb)
        return DebtTangibleEquityRatio

    def TangibleAToInteBearDebt(self):
        """
        因子描述：
        有形净值/带息负债。
        计算方法：
        TangibleAToInteBearDebt = 有形净值/带息负债
        注：财务科目的空值用前值填充
        """
        FixAssets = self.df_bic["fix_assets"].fillna(method="pad").fillna(0)
        cip = self.df_bic["cip"].fillna(method="pad").fillna(0)
        NTanAssets = FixAssets + cip
        St_B = self.df_bic["st_borr"].fillna(method="pad")
        non_Cur_l = self.df_bic["non_cur_liab_due_1y"].fillna(method="pad").fillna(0)
        LT_debt = self.df_bic["lt_borr"].fillna(method="pad").fillna(0)
        Bond_payable = self.df_bic["bond_payable"].fillna(method="pad").fillna(0)
        other_payable = self.df_bic["oth_payable"].fillna(method="pad").fillna(0)
        InterestBearDebt = St_B + non_Cur_l + LT_debt + Bond_payable + other_payable
        ta2bd = NTanAssets / InterestBearDebt
        TA2BD = self.drop_duplicated_ann_date(ta2bd)
        TangibleAToInteBearDebt = self.expand_index(TA2BD, self.df_sb)
        return TangibleAToInteBearDebt

    def TangibleAToNetDebt(self):
        """
        因子描述：
        有形净值/净债务。
        计算方法：
        TangibleAToNetDebt=有形净值/净债务
        注：财务科目的空值用前值填充
        净债务=总债务-现金及现金等价物期末余额
        """
        FixAssets = self.df_bic["fix_assets"].fillna(method="pad")
        cip = self.df_bic["cip"].fillna(method="pad").fillna(0)
        NTanAssets = FixAssets + cip
        total_debt = self.df_bic["total_liab"].fillna(method="pad")
        cash_eqv = self.df_bic["c_cash_equ_end_period"].fillna(method="pad").fillna(0)
        NetDebt = total_debt - cash_eqv
        TA2ND = NTanAssets / NetDebt
        tangibleAToNetDebt = self.drop_duplicated_ann_date(TA2ND)
        TangibleAToNetDebt = self.expand_index(tangibleAToNetDebt, self.df_sb)
        return TangibleAToNetDebt

    def NOCFToTLiability(self):
        """
        因子描述：
        经营活动产生的现金流量净额/负债合计。
        计算方法：NOCFToTLiability=经营活动产生的现金流量净额(NOCF)/负债合计(TL) 计算方法；NOCF为TTM，TL为latest
        """
        NOCF = self.df_bic["n_cashflow_act"].rolling(4, 1).mean().fillna(0)
        TL = self.df_bic["total_liab"].fillna(0)
        NOCF2TLiability = NOCF / TL
        NOC = self.drop_duplicated_ann_date(NOCF2TLiability)
        NOCFToTLiability = self.expand_index(NOC, self.df_sb)
        return NOCFToTLiability

    def NOCFToInterestBearDebt(self):
        """
        因子描述：
        经营活动产生现金流量净额/带息负债。
        计算方法：
        NOCFToInterestBearDebt = 经营活动产生现金流量净额TTM / 带息负债
        """
        net_CF = self.df_bic["n_cashflow_act"].rolling(4, 1).mean().fillna(0)
        St_B = self.df_bic["st_borr"].fillna(method="pad").fillna(0)
        non_Cur_l = self.df_bic["non_cur_liab_due_1y"].fillna(method="pad").fillna(0)
        LT_debt = self.df_bic["lt_borr"].fillna(method="pad").fillna(0)
        Bond_payable = self.df_bic["bond_payable"].fillna(method="pad").fillna(0)
        other_payable = self.df_bic["oth_payable"].fillna(method="pad").fillna(0)
        InterestBearDebt = St_B + non_Cur_l + LT_debt + Bond_payable + other_payable
        NOCF2IBT = net_CF / InterestBearDebt
        NOCF2InterestBearDebt = self.drop_duplicated_ann_date(NOCF2IBT)
        NOCFToInterestBearDebt = self.expand_index(NOCF2InterestBearDebt, self.df_sb)
        return NOCFToInterestBearDebt

    def NOCFToNetDebt(self):
        """
        因子描述：
        经营活动产生现金流量净额/净债务。
        计算方法：
        NOCFToNetDebt = 经营活动产生现金流量净额TTM / 净债务
        """
        Net_CF_act = self.df_bic["n_cashflow_act"].rolling(4, min_periods=0).mean().fillna(0)
        total_debt = self.df_bic["total_liab"].fillna(method="pad")
        cash_eqv = self.df_bic["c_cash_equ_end_period"].fillna(method="pad")
        NetDebt = total_debt - cash_eqv
        NOCF2ND = Net_CF_act / NetDebt
        NOCFtoND = self.drop_duplicated_ann_date(NOCF2ND)
        NOCFToNetDebt = self.expand_index(NOCFtoND, self.df_sb)
        return NOCFToNetDebt

    def TSEPToTotalCapital(self):
        """
        因子描述：
        归属于母公司所有者权益合计/全部投入资本。
        计算方法：
        TSEPToTotalCapital=归属于母公司所有者权益合计/全部投入资本
        """

    def InteBearDebtToTotalCapital(self):
        """
        因子描述：
        带息负债/全部投入资本。
        计算方法：
        InteBearDebtToTotalCapital=带息负债(InterestBearDebt)/全部投入资本(TotalCapital)  计算方法均为latest
        """

