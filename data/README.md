# Data

The raw data is not committed: the game logs are 159 MB (over GitHub's file limit) and the odds dataset has no stated license. Download both into `data/raw/`:

| File | Source | License |
|---|---|---|
| `data/raw/game_logs.csv` | [Major League Baseball Game Logs](https://www.kaggle.com/datasets/thedevastator/major-league-baseball-game-logs) — Dataquest compilation of [Retrosheet](https://www.retrosheet.org/gamelogs/index.html) game logs, 1871–2016 | CC BY-SA 4.0 |
| `data/raw/oddsDataMLB.csv` | [MLB Odds Data](https://www.kaggle.com/datasets/christophertreasure/major-league-baseball-vegas-data) — closing moneyline, run line and total, 2012–2021 | Not stated ("Other") |

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configured:

```bash
kaggle datasets download -d thedevastator/major-league-baseball-game-logs -p data/raw --unzip
kaggle datasets download -d christophertreasure/major-league-baseball-vegas-data -p data/raw --unzip
```

Or download each from the Kaggle page and place the CSVs in `data/raw/` with the names above.

## Notes

- Only the overlapping 2012–2016 regular seasons are used (12,148 games; 12,146 matched to closing odds).
- Retrosheet notice: *The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org.*
