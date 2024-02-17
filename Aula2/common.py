import sqlite3
import pandas as pd

class Settings:
    def __init__(self, **kargs) -> None:
        self.dbname = kargs.get('dbname', './colabfilter.db')
        self.item_column = kargs.get('item_column', 'WineID')
        self.user_column = kargs.get('user_column', 'UserID')
        self.rating_column = kargs.get('rating_column', 'Rating')
        
 
settings_simple = Settings(dbname='./simple-sim.db', item_column='ItemID')

settings_winex = Settings(dbname='./winex-sim.db')

class SimItem:
    ''' Classe para representar os índices da tabela de similaridade criada no fcwinex-offline.py
    '''
    TIMESTAMP = 0
    SOURCE = 1
    TARGET = 2
    SIM = 3


def get_conn(settings):
    conn = sqlite3.connect(settings.dbname)
    return conn
    
def load_winex_ratings(min_ratings=1):
    columns = ['UserID', 'WineID', 'Rating', 'type']
    #wine_data = pd.read_csv()
    ratings = pd.read_csv('../data/XWines_Slim_150K_ratings.csv', low_memory=False)
    ratings.head()
    user_count = ratings[['UserID', 'WineID']].groupby('UserID').count()
    user_count = user_count.reset_index()
    UserIDs = user_count[user_count['WineID'] > min_ratings]['UserID']
    ratings = ratings[ratings['UserID'].isin(UserIDs)]
    ratings['Rating'] = ratings['Rating'].astype(float)
    return ratings

