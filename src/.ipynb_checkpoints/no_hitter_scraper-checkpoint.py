import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

response = requests.get('https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_no-hitters')

# Scrape the wikipedia table corresponding to no-hitters
no_hit_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_no-hitters')[1]

# for demonstration, look at games from 1998, which all had 162 games for 30 teams. The game count is approximately 2430 games
no_hit_table['Date'] = pd.to_datetime(no_hit_table['Date'])
no_hit_table['Year'] = no_hit_table.Date.apply(lambda x: x.year)

# Don't count 1998 given it is currently under way.
no_hit_table_1998= no_hit_table[(no_hit_table["Date"] >= "1998") & (no_hit_table["Date"] < "2020")]

# Total games multiplied by the number of years from 98 - 2019
total_games = 2430 * no_hit_table_1998.groupby('Year').count().shape[0]

# Number of no-hitters is just the number of records in the table: shape[0]
# Probability of a nohitter in a game is that number divided by the total number of games played over all the seasons.
prob_of_nohit = no_hit_table_1998.shape[0]/total_games

x = np.arange(stats.binom.ppf(.00001, 2430, prob_of_nohit),
          stats.binom.ppf(.99999, 2430, prob_of_nohit))

fig, ax = plt.subplots()
ax.bar(x, stats.binom.pmf(x, 2430, prob_of_nohit))

