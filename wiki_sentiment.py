#update this to read the existing file and then only add on the ones that are missing
def main():
    print('Getting wikipedia data...')
    import os
    import pandas as pd
    
    import mwclient
    import time
    site = mwclient.Site("en.wikipedia.org")
    page = site.pages["Bitcoin"]

    revs = list(page.revisions())
    revs = sorted(revs,key = lambda rev: rev["timestamp"])
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    def find_sentiment(text):
        sent = sentiment_pipeline([text[:250]])[0]
        score = sent['score']
        if sent['label'] == "NEGATIVE":
            score *=-1
        return score

    edits = {}
    print('Parsing revisions....')
    for rev in revs:
        date = time.strftime("%Y-%m-%d", rev["timestamp"])
        if date not in edits:
            edits[date]=dict(sentiments = list(), edit_count = 0)
        edits[date]["edit_count"] +=1
        try:
            comment = rev['comment']
        except: 
            comment = ''
        edits[date]['sentiments'].append(find_sentiment(comment))
        
    from statistics import mean
    print("Getting Sentiment Scores...")
    for key in edits:
        if(len(edits[key]['sentiments'])>0):
            edits[key]['sentiment'] = mean(edits[key]['sentiments'])
            edits[key]['neg_sentiment'] = len ([s for s in edits[key]['sentiments'] if s < 0]) / len(edits[key]['sentiments'])
        else:
            edits[key]['sentiment'] = 0
            edits[key]['neg_sentiments'] = 0
        del edits[key]['sentiments']
        
    import pandas as pd
    edits_df = pd.DataFrame.from_dict(edits, orient = 'index')
    edits_df.index = pd.to_datetime(edits_df.index, utc = True)
    from datetime import datetime


        
    dates = pd.date_range(start = '2009-03-08', end = datetime.utcnow().date())
    edits_df = edits_df.reindex(dates, fill_value = 0)
    rolling_edits = edits_df.rolling(30).mean()
    rolling_edits = rolling_edits.dropna()

    return rolling_edits

if(__name__ == '__main__'):
    import os
    rolling_edits = main()
    rolling_edits.to_csv('wikipedia_edits.csv')
    
