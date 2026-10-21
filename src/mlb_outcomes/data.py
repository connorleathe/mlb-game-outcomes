"""
Load the Retrosheet game logs and the sportsbook odds, and merge them into one
row per game with vig-removed market probabilities.
"""

import numpy as np
import pandas as pd

RETRO_TO_ABBR = {
    'LAN':'LAD','SFN':'SF','SDN':'SD','COL':'COL','ARI':'ARI',
    'CHN':'CHC','SLN':'STL','CIN':'CIN','MIL':'MIL','PIT':'PIT',
    'NYN':'NYM','PHI':'PHI','ATL':'ATL','MIA':'MIA','FLA':'MIA','WAS':'WSH','MON':'WSH',
    'ANA':'LAA','LAA':'LAA','OAK':'OAK','SEA':'SEA','TEX':'TEX','HOU':'HOU',
    'CHA':'CWS','CLE':'CLE','DET':'DET','KCA':'KC','MIN':'MIN',
    'NYA':'NYY','BOS':'BOS','TBA':'TBR','TBD':'TBR','BAL':'BAL','TOR':'TOR',
}
MLB_ABBR_TO_FULL = {
    'NYY':'New York Yankees','Yankees':'New York Yankees',
    'BOS':'Boston Red Sox','Red Sox':'Boston Red Sox',
    'TBR':'Tampa Bay Rays','TB':'Tampa Bay Rays','Rays':'Tampa Bay Rays',
    'BAL':'Baltimore Orioles','Orioles':'Baltimore Orioles',
    'TOR':'Toronto Blue Jays','Blue Jays':'Toronto Blue Jays',
    'CLE':'Cleveland Guardians','Guardians':'Cleveland Guardians','Cleveland Indians':'Cleveland Guardians',
    'CWS':'Chicago White Sox','White Sox':'Chicago White Sox',
    'DET':'Detroit Tigers','Tigers':'Detroit Tigers',
    'KC':'Kansas City Royals','KCR':'Kansas City Royals','Royals':'Kansas City Royals',
    'MIN':'Minnesota Twins','Twins':'Minnesota Twins',
    'HOU':'Houston Astros','Astros':'Houston Astros',
    'LAA':'Los Angeles Angels','Angels':'Los Angeles Angels',
    'OAK':'Oakland Athletics','Athletics':'Oakland Athletics',"A's":'Oakland Athletics','As':'Oakland Athletics',
    'SEA':'Seattle Mariners','Mariners':'Seattle Mariners',
    'TEX':'Texas Rangers','Rangers':'Texas Rangers',
    'ATL':'Atlanta Braves','Braves':'Atlanta Braves',
    'MIA':'Miami Marlins','FLA':'Miami Marlins','Marlins':'Miami Marlins',
    'NYM':'New York Mets','Mets':'New York Mets',
    'PHI':'Philadelphia Phillies','Phillies':'Philadelphia Phillies',
    'WSH':'Washington Nationals','WAS':'Washington Nationals','Nationals':'Washington Nationals',
    'CHC':'Chicago Cubs','Cubs':'Chicago Cubs',
    'CIN':'Cincinnati Reds','Reds':'Cincinnati Reds',
    'MIL':'Milwaukee Brewers','Brewers':'Milwaukee Brewers',
    'PIT':'Pittsburgh Pirates','Pirates':'Pittsburgh Pirates',
    'STL':'St. Louis Cardinals','Cardinals':'St. Louis Cardinals',
    'ARI':'Arizona Diamondbacks','Dbacks':'Arizona Diamondbacks','Diamondbacks':'Arizona Diamondbacks',
    'COL':'Colorado Rockies','Rockies':'Colorado Rockies',
    'LAD':'Los Angeles Dodgers','LA Dodgers':'Los Angeles Dodgers','Dodgers':'Los Angeles Dodgers',
    'SD':'San Diego Padres','SDP':'San Diego Padres','Padres':'San Diego Padres',
    'SF':'San Francisco Giants','SFG':'San Francisco Giants','Giants':'San Francisco Giants',
}


def _to_full(name: str) -> str:
    s = str(name).strip()
    return MLB_ABBR_TO_FULL.get(s.upper(), MLB_ABBR_TO_FULL.get(s.title(), s.title()))


def _retro_to_full(code: str) -> str:
    s = str(code).strip().upper()
    abbr = RETRO_TO_ABBR.get(s, s)
    return _to_full(abbr)


def _parse_any_date(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip()
    d = pd.to_datetime(s, format='%Y%m%d', errors='coerce')
    d = d.fillna(pd.to_datetime(s, errors='coerce'))
    return d


def moneyline_to_prob(ml):
    ml = float(ml)
    return (-ml / (-ml + 100.0)) if ml < 0 else (100.0 / (ml + 100.0))


def load_and_clean_gamelogs(path) -> pd.DataFrame:
    """Retrosheet game logs -> one row per game with scores and outcome targets."""
    df = pd.read_csv(path, low_memory=False)
    colmap = {}
    if 'h_name' in df.columns:  colmap['h_name']  = 'home_team'
    if 'v_name' in df.columns:  colmap['v_name']  = 'away_team'
    if 'h_score' in df.columns: colmap['h_score'] = 'home_score'
    if 'v_score' in df.columns: colmap['v_score'] = 'away_score'
    df = df.rename(columns=colmap)

    need = ['date','home_team','away_team','home_score','away_score']
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"Game logs missing required columns: {miss}")

    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)
    df['home_margin'] = df['home_score'] - df['away_score']
    df['abs_margin']  = df['home_margin'].abs()
    df['home_win']    = (df['home_margin'] > 0).astype(int)
    return df[['date','home_team','away_team','home_score','away_score','home_margin','abs_margin','home_win']]


def load_odds_teamrows(path) -> pd.DataFrame:
    """Odds file -> one row per team per game (closing moneyline, run line, total, final score)."""
    raw = pd.read_csv(path, low_memory=False).rename(columns={
        'Date':'date','Team':'team','Opponent':'opponent',
        'MoneyLine':'moneyLine','RunLine':'runLine','Total':'total'
    })
    if not {'date','team','opponent','runs','oppRuns'}.issubset(raw.columns):
        raise ValueError(f"Odds file must include date/team/opponent/runs/oppRuns; got: {list(raw.columns)[:12]}")

    return pd.DataFrame({
        'date'     : _parse_any_date(raw['date']),
        'team'     : raw['team'].apply(_to_full),
        'opponent' : raw['opponent'].apply(_to_full),
        'runs'     : pd.to_numeric(raw['runs'], errors='coerce'),
        'oppRuns'  : pd.to_numeric(raw['oppRuns'], errors='coerce'),
        'moneyLine': pd.to_numeric(raw.get('moneyLine'), errors='coerce'),
        'runLine'  : pd.to_numeric(raw.get('runLine'),  errors='coerce'),
        'total'    : pd.to_numeric(raw.get('total'),    errors='coerce'),
    })


def merge_games_and_odds(g: pd.DataFrame, odds_team: pd.DataFrame) -> pd.DataFrame:
    """
    Attach each game's own closing line.

    Joins on date + team + opponent + final score. Joining on date + team alone
    cross-joins the two games of a doubleheader with both games' odds (4 rows
    per doubleheader, half of them with the wrong line); the final score
    disambiguates them. Games that cannot be matched keep NaN odds.
    """
    odds_start, odds_end = odds_team['date'].min(), odds_team['date'].max()
    g = g[(g['date'] >= odds_start) & (g['date'] <= odds_end)].copy()
    g['home_team'] = g['home_team'].apply(_retro_to_full)
    g['away_team'] = g['away_team'].apply(_retro_to_full)

    # A doubleheader with identical final scores can't be told apart: drop those lines.
    keys = ['date','team','opponent','runs','oppRuns']
    odds_team = odds_team.drop_duplicates(subset=keys, keep=False)

    home = odds_team.rename(columns={
        'team':'home_team','opponent':'away_team','runs':'home_score','oppRuns':'away_score',
        'moneyLine':'home_moneyline','total':'closing_total','runLine':'home_runline'})
    away = odds_team.rename(columns={
        'team':'away_team','opponent':'home_team','runs':'away_score','oppRuns':'home_score',
        'moneyLine':'away_moneyline'})[['date','home_team','away_team','home_score','away_score','away_moneyline']]

    game_keys = ['date','home_team','away_team','home_score','away_score']
    df = (g.merge(home[game_keys + ['home_moneyline','closing_total','home_runline']], on=game_keys, how='left')
           .merge(away, on=game_keys, how='left'))
    df['closing_runline'] = df.pop('home_runline').abs()
    assert len(df) == len(g), "merge must not duplicate games"
    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implied probabilities from moneylines with the bookmaker's overround (vig) removed."""
    df = df.copy()
    df['home_prob_raw'] = df['home_moneyline'].apply(lambda x: moneyline_to_prob(x) if pd.notna(x) else np.nan)
    df['away_prob_raw'] = df['away_moneyline'].apply(lambda x: moneyline_to_prob(x) if pd.notna(x) else np.nan)
    df['overround'] = df['home_prob_raw'] + df['away_prob_raw']
    df['home_prob'] = df['home_prob_raw'] / df['overround']
    df['away_prob'] = df['away_prob_raw'] / df['overround']
    df['fav_edge']  = (df['home_prob'] - df['away_prob']).abs()
    df['home_is_fav'] = (df['home_prob'] > df['away_prob'])
    return df


def load_dataset(path_gamelogs, path_odds) -> pd.DataFrame:
    g = load_and_clean_gamelogs(path_gamelogs)
    odds_team = load_odds_teamrows(path_odds)
    return add_market_features(merge_games_and_odds(g, odds_team))
