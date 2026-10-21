"""
Pre-game features built only from the game logs: Elo ratings, park factor,
rest / back-to-back / venue switches, and last-3-days run scoring.
"""

import numpy as np
import pandas as pd


def add_elo(df_in, k=18.0, home_adv=52.0):
    """Per-season Elo (reset to 1500 each season). Ratings are recorded BEFORE each game is played."""
    df_e = df_in.copy()
    df_e[['elo_home','elo_away','delta_elo']] = np.nan
    for season, chunk in df_e.groupby(df_e['date'].dt.year, sort=True):
        teams = pd.unique(chunk[['home_team','away_team']].values.ravel('K'))
        rating = {t: 1500.0 for t in teams}
        for i in chunk.index:
            h, a = df_e.at[i,'home_team'], df_e.at[i,'away_team']
            hs, as_ = df_e.at[i,'home_score'], df_e.at[i,'away_score']
            Rh, Ra = rating.get(h,1500.0), rating.get(a,1500.0)
            df_e.at[i,'elo_home'] = Rh
            df_e.at[i,'elo_away'] = Ra
            df_e.at[i,'delta_elo'] = Rh - Ra
            if pd.isna(hs) or pd.isna(as_):  # safety
                continue
            Rh_adj = Rh + home_adv
            Eh = 1.0 / (1.0 + 10 ** ((Ra - Rh_adj)/400.0))
            Ea = 1.0 - Eh
            sh, sa = int(hs > as_), int(as_ > hs)
            rating[h] = Rh + k * (sh - Eh)
            rating[a] = Ra + k * (sa - Ea)
    return df_e


def add_park_factor(df):
    """Seasonal park factor: home park runs/game divided by league runs/game (same season)."""
    tmp = df.copy()
    tmp['runs'] = tmp['home_score'] + tmp['away_score']
    season_avg = tmp.groupby('season')['runs'].mean().rename('lg_rpg')
    home_avg   = tmp.groupby(['season','home_team'])['runs'].mean().rename('home_rpg')
    pf = (home_avg.reset_index().merge(season_avg.reset_index(), on='season', how='left'))
    pf['park_factor'] = pf['home_rpg'] / pf['lg_rpg']
    return df.merge(pf[['season','home_team','park_factor']], on=['season','home_team'], how='left')


def compute_rest_features_long(df_in: pd.DataFrame) -> pd.DataFrame:
    """Rest days, back-to-back flag and home/away venue switch for each team."""
    # Preserve row ids to merge back
    base = df_in[['date','home_team','away_team']].copy()
    base['row_id'] = base.index

    # Build long schedule: one row per team-game
    home_long = base[['row_id','date','home_team']].rename(columns={'home_team':'team'})
    home_long['is_home'] = 1
    away_long = base[['row_id','date','away_team']].rename(columns={'away_team':'team'})
    away_long['is_home'] = 0
    long = pd.concat([home_long, away_long], ignore_index=True)

    # Sort by team/date and compute rest metrics
    long = long.sort_values(['team','date'])
    long['prev_date'] = long.groupby('team')['date'].shift(1)
    long['rest_days'] = (long['date'] - long['prev_date']).dt.days
    long['prev_is_home'] = long.groupby('team')['is_home'].shift(1)
    long['venue_switch'] = (long['prev_is_home'] != long['is_home']).astype('int')
    long['b2b'] = (long['rest_days'] == 1).astype('int')

    # Split back to home/away and merge to original rows via row_id
    home_feats = (long[long['is_home']==1]
                  .set_index('row_id')[['rest_days','b2b','venue_switch']]
                  .rename(columns={'rest_days':'home_rest_days',
                                   'b2b':'home_b2b',
                                   'venue_switch':'home_venue_switch'}))

    away_feats = (long[long['is_home']==0]
                  .set_index('row_id')[['rest_days','b2b','venue_switch']]
                  .rename(columns={'rest_days':'away_rest_days',
                                   'b2b':'away_b2b',
                                   'venue_switch':'away_venue_switch'}))

    out = df_in.copy()
    out = out.join(home_feats, how='left').join(away_feats, how='left')
    return out


def last_k_days_sum(df_in, team_col, score_col, k_days=3, label=''):
    """Runs in the team's games over the previous k days (strictly before game day)."""
    g = df_in[['date', team_col, score_col]].copy().sort_values(['date'])
    out = pd.Series(index=df_in.index, dtype='float')
    for t, sub in g.groupby(team_col):
        idx = sub.index
        d = sub['date']
        s = sub[score_col].fillna(0)
        vals = []
        for i in range(len(sub)):
            start = d.iloc[i] - pd.Timedelta(days=k_days)
            mask = (d > start) & (d < d.iloc[i])  # strictly before same day
            vals.append(s[mask].sum())
        out.loc[idx] = vals
    out.name = label
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values('date').reset_index(drop=True)
    df['season'] = df['date'].dt.year
    df = add_elo(df, k=18.0, home_adv=52.0)
    df = add_park_factor(df)
    df = compute_rest_features_long(df)
    df['home_last3_off'] = last_k_days_sum(df, 'home_team', 'home_score', 3, 'home_last3_off')
    df['home_last3_def'] = last_k_days_sum(df, 'home_team', 'away_score', 3, 'home_last3_def')
    df['away_last3_off'] = last_k_days_sum(df, 'away_team', 'away_score', 3, 'away_last3_off')
    df['away_last3_def'] = last_k_days_sum(df, 'away_team', 'home_score', 3, 'away_last3_def')
    return df
