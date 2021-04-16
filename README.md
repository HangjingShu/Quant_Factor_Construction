The first step to construct my own quantitative framework, also a part of outcome of my internship. The reference of the construction approach is from datayes, the data is from Wind and Tushares and stored in Navicat.

Some instruction: Since the data is extract from database by sqlalchemy and logging, I delete the related code and file to avoid misunderstanding, just need to know:

1. bic: The information of financial report, which merged balance sheet, income statement and cash flow. Of course, we did lots of data cleaning and reshaping, but I choose not to show it in public
2. sb: stock_basic, the indirect data of stock like PE, market capitalization
3. ac: the predicted value of EPS, income, profit from analyst
4. OHLC: In Chinese, 股票后复权数据，like close, open, high, low of stock price and trade volume.
5. OHLC_qfq：股票前复权数据

The inspiration to build my unique AnalystConsensus factor is the research paper of China securities.
