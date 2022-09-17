import twint

c = twint.Config()
c.Username = "elonmusk"
c.Pandas = True
c.Retweets = True
twint.run.Search(c)

Tweets_df = twint.storage.panda.Tweets_df
Tweets_df.to_csv("elonmusk.csv", index = None)

# on cmd prompt, run:
# python twintElonMuskBitcoin.py