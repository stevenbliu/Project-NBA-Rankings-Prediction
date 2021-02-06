from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import torch
import numpy as np


def get_features(season):
    features = pd.read_csv('data/features/feat' + str(season) + '.csv')

    features = features.set_index(features['Tm'].map(id2idx)).drop(columns=['Tm'])
    features = features.sort_index()

    labels = torch.Tensor(features['Rk'].to_numpy())
    features = torch.Tensor(features.drop(columns='Rk').to_numpy())

    return features, labels

def get_adjacency(season):

    sc = pd.read_csv('data/schedule/sch' + str(season))

    adj = np.zeros((30, 30), dtype='float32')

    rows = sc['Home']
    cols = sc['Away']

    for i in np.arange(len(sc)):
        adj[ id2idx[rows[i]], id2idx[cols[i]]] = adj[ id2idx[rows[i]], id2idx[cols[i]]] + 1.0

    adj = torch.Tensor(adj)

    return adj

def get_season_ranks(season):
    rankings = pd.read_csv('data/ranks/rank{}.csv'.format(season))
    rankings['Team'] = rankings['Team'].apply(lambda x: abbrev[x.upper()])
    rankings = rankings.set_index('Team').drop(columns=['Overall'])
    return rankings


def get_season_stats(season):
    # NBA season we will be analyzing
    year = season
    # URL page we will scraping (see image above)
    url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html".format(year)
    # this is the HTML from the given URL
    html = urlopen(url)
    soup = BeautifulSoup(html)

    #organize it into a list:
    # use findALL() to get the column headers
    soup.findAll('tr', limit=2)
    # use getText()to extract the text we need into a list
    headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]
    # exclude the first column as we will not need the ranking order from Basketball Reference for the analysis
    headers = headers[1:]


    # avoid the first header row
    rows = soup.findAll('tr')[1:]
    player_stats = [[td.getText() for td in rows[i].findAll('td')]
                for i in range(len(rows))]

    stats = pd.DataFrame(player_stats, columns = headers)
    return stats


abbrev = {'ATLANTA HAWKS' : 'ATL',
        'ST. LOUIS HAWKS' : 'SLH',
        'MILWAUKEE HAWKS' : 'MIL',
        'TRI-CITIES BLACKHAWKS' : 'TCB',
        'BOSTON CELTICS' : 'BOS',
        'BROOKLYN NETS' : 'BRK',
        'NEW JERSEY NETS' : 'NJN',
        'CHICAGO BULLS' : 'CHI',
        'CHARLOTTE HORNETS (1988-2004)': 'CHH',
        'CHARLOTTE HORNETS (2014-Present)': 'CHO',
        'CHARLOTTE HORNETS': 'CHO',
        'CHARLOTTE BOBCATS' : 'CHA',
        'CLEVELAND CAVALIERS' : 'CLE',
        'DALLAS MAVERICKS': 'DAL',
        'DENVER NUGGETS' : 'DEN',
        'DETROIT PISTONS' : 'DET',
        'FORT WAYNE PISTONS' : 'FWP',
        'GOLDEN STATE WARRIORS' : 'GSW',
        'SAN FRANCISCO WARRIORS' : 'SFW',
        'PHILADELPHIA WARRIORS' : 'PHI',
        'HOUSTON ROCKETS' : 'HOU',
        'INDIANA PACERS' : 'IND',
        'LOS ANGELES CLIPPERS' : 'LAC',
        'SAN DIEGO CLIPPERS' : 'SDC',
        'BUFFALO BRAVES' : 'BUF',
        'LOS ANGELES LAKERS' : 'LAL',
        'MINNEAPOLIS LAKERS' : 'MIN',
        'MEMPHIS GRIZZLIES' : 'MEM',
        'VANCOUVER GRIZZLIES' : 'VAN',
        'MIAMI HEAT' : 'MIA',
        'MILWAUKEE BUCKS' : 'MIL',
        'MINNESOTA TIMBERWOLVES' : 'MIN',
        'NEW ORLEANS PELICANS' : 'NOP',
        'NEW ORLEANS/OKLAHOMA CITY HORNETS' : 'NOK',
        'NEW ORLEANS HORNETS' : 'NOH',
        'NEW YORK KNICKS' : 'NYK',
        'OKLAHOMA CITY THUNDER' : 'OKC',
        'SEATTLE SUPERSONICS' : 'SEA',
        'ORLANDO MAGIC' : 'ORL',
        'PHILADELPHIA 76ERS' : 'PHI',
        'SYRACUSE NATIONALS' : 'SYR',
        'PHOENIX SUNS' : 'PHO',
        'PORTLAND TRAIL BLAZERS' : 'POR',
        'SACRAMENTO KINGS' : 'SAC',
        'KANSAS CITY KINGS' : 'KCK',
        'KANSAS CITY-OMAHA KINGS' : 'KCK',
        'CINCINNATI ROYALS' : 'CIN',
        'ROCHESTER ROYALS' : 'ROR',
        'SAN ANTONIO SPURS' : 'SAS',
        'TORONTO RAPTORS' : 'TOR',
        'UTAH JAZZ' : 'UTA',
        'NEW ORLEANS JAZZ' : 'NOJ',
        'WASHINGTON WIZARDS' : 'WAS',
        'WASHINGTON BULLETS' : 'WAS',
        'CAPITAL BULLETS' : 'CAP',
        'BALTIMORE BULLETS' : 'BAL',
        'CHICAGO ZEPHYRS' : 'CHI',
        'CHICAGO PACKERS' : 'CHI',
        'ANDERSON PACKERS' : 'AND',
        'CHICAGO STAGS' : 'CHI',
        'INDIANAPOLIS OLYMPIANS' : 'IND',
        'SHEBOYGAN RED SKINS' : 'SRS',
        'ST. LOUIS BOMBERS' : 'SLB',
        'WASHINGTON CAPITOLS' : 'WAS',
        'WATERLOO HAWKS' : 'WAT'
        }

id2idx = {'PHO': 0,
         'DAL': 1,
         'POR': 2,
         'OKC': 3,
         'DEN': 4,
         'MEM': 5,
         'WAS': 6,
         'MIA': 7,
         'BRK': 8,
         'CLE': 9,
         'TOR': 10,
         'NOP': 11,
         'HOU': 12,
         'IND': 13,
         'LAC': 14,
         'PHI': 15,
         'SAC': 16,
         'UTA': 17,
         'LAL': 18,
         'BOS': 19,
         'ORL': 20,
         'MIL': 21,
         'SAS': 22,
         'ATL': 23,
         'GSW': 24,
         'CHI': 25,
         'NYK': 26,
         'DET': 27,
         'MIN': 28,
         'CHO': 29
        }
