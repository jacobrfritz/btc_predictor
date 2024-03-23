import yfinance as yf
import os
import pandas as pd

def predict(train,test,predictors,model):
    model.fit(train[predictors], train['target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds,index=test.index, name = 'predictions')
    combined = pd.concat([test['target'],preds], axis = 1)
    return combined

def backtest(data,model,predictors, start = 1095, step = 150):
    all_predictions = []
    for i in range(start,data.shape[0],step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def compute_rolling(btc):
    horizons = [2,7,30,180]
    new_predictors = ['close','volume','open','low','edit_count','sentiment','neg_sentiment','coins_per_block','sp_close','sp_volume','sp_open','sp_low','sp_high']
    
    for horizon in horizons:
        rolling_averages = btc.rolling(horizon, min_periods = 1).mean()

        ratio_column = f'close_ratio_{horizon}'
        btc[ratio_column] = btc['close']/ rolling_averages['close']

        edit_column = f'edit_{horizon}'
        btc[edit_column] = rolling_averages['edit_count']

        rolling = btc.rolling(horizon, closed = 'left',min_periods=1).mean()
        trend_column = f'trend_{horizon}'
        btc[trend_column] = rolling['target']
        
        sp_trend_column = f'sp_close_ratio_{horizon}'
        btc[sp_trend_column] = rolling['sp_close']
        
        #cpb_trend_column = f'cpb_trend_column_{horizon}'
        #btc[cpb_trend_column] = rolling['coins_per_block']
        
        new_predictors += [ratio_column,trend_column,edit_column,sp_trend_column]
    return btc, new_predictors

def main():
    from datetime import datetime
    
    btc_ticker = yf.Ticker("BTC-USD")
    btc = btc_ticker.history(period="max")
    
    btc.index = pd.to_datetime(btc.index, utc = True)

    btc = btc.reset_index()

    btc['Date'] = pd.to_datetime(btc['Date'], utc = True)
    #need to delete this part once yfinance gets fixed
    #its correcting the fact that it is going from mar 20 to mar 22???
    today = datetime.today().strftime("%Y-%m-%d")
    now_time = datetime.strptime(today+ ' 00:00:00+00:00',"%Y-%m-%d %H:%M:%S%z")
    btc.loc[len(btc)-1, "Date"] = now_time

    btc.set_index(btc['Date'],inplace = True) 
    del btc['Date']
    btc.index = pd.to_datetime(btc.index, utc = True)

    sp_ticker = yf.Ticker("^GSPC")
    sp = sp_ticker.history(period="max")
    
    sp.index = pd.to_datetime(sp.index, utc= True).date
    sp.index = pd.to_datetime(sp.index, utc= True)
 
    
    #stock splits left off S&p data??? wtf is going on here
    #DELETE

    del btc['Dividends']
    del btc['Stock Splits']
    del sp['Dividends']
    del sp['Stock Splits']

    btc.columns = [c.lower() for c in btc.columns]
    sp.columns = [f'sp_{c.lower()}' for c in sp.columns]
    
    import wiki_sentiment
    wiki_sentiment.main()
    wiki =  pd.read_csv('wikipedia_edits.csv', index_col = 0, parse_dates = True)
    wiki.index = pd.to_datetime(wiki.index,utc = True)
    
    print('ETL....')

    btc = btc.join(sp, how = 'left')

    btc = btc.join(wiki)
    

    btc['tomorrow'] = btc['close'].shift(-1)
    btc['target'] = (btc['tomorrow'] >btc['close']).astype(int)

    fifty_btc = pd.DataFrame(pd.date_range('2009-01-03','2012-11-27'))
    fifty_btc['coins_per_block'] = 50
    twenty_five = pd.DataFrame(pd.date_range('2012-11-28', '2016-07-08'))
    twenty_five['coins_per_block'] = 25
    twelve_five_btc = pd.DataFrame(pd.date_range('2016-07-09', '2020-05-10'))
    twelve_five_btc['coins_per_block'] = 12.5
    six_two_five_btc = pd.DataFrame(pd.date_range('2020-05-11', '2024-04-19'))
    six_two_five_btc['coins_per_block'] = 6.25
    three_one_two_five_btc = pd.DataFrame(pd.date_range('2024-04-20', datetime.utcnow().date()))
    three_one_two_five_btc['coins_per_block'] = 3.125
    fifty_btc
    coins_per_block = pd.concat([fifty_btc,twenty_five,twelve_five_btc, six_two_five_btc,three_one_two_five_btc], axis = 0)
    
    coins_per_block.index = pd.to_datetime(coins_per_block[0], utc = True)
    del coins_per_block[0]
    
    btc = btc.join(coins_per_block)

    
    print('Generating Features...')
    
    btc, predictors = compute_rolling(btc.copy())
    print('Backtesting....')
    from sklearn.metrics import precision_score
    
   #print(btc[[col for col in btc.columns if 'sp' in col]])
    from xgboost import XGBClassifier
    
    model = XGBClassifier(random_state=1,learning_rate=.1,n_estimators=200)
    predictions = backtest(btc, model,predictors)

    precision_score = precision_score(predictions['target'], predictions['predictions'])
    predictions.to_csv('predictions.csv')
    print(btc)
    print(f'Precision Score: {precision_score}')
    return predictions

if __name__ == '__main__':
    print(main())
