# UFC Fight Prediction Model

# --------------------
# Libraries
# --------------------

import pandas as pd
import numpy as np
import joblib

# --------------------
# Load Artifacts
# --------------------

PREP_PATH = "artifacts/preprocessing_pipeline.joblib"
MODEL_PATH = "artifacts/logistic_model.joblib"

preprocess = joblib.load(PREP_PATH)
model = joblib.load(MODEL_PATH)

# --------------------
# Define Feature Engineering Functions
# --------------------

# Define column order
COL_ORDER = [
    # Fight metadata
    "FightID", "RedFighter", "BlueFighter", "Date", "Year", "Country", "Gender", "TitleBout", "WeightClass", "NumberOfRounds",
    
    # Fighter stats
    "IsRedDebut", "IsBlueDebut", "RedDaysSinceLastFight", "BlueDaysSinceLastFight",
    "RedWins", "BlueWins", "RedLosses", "BlueLosses", "RedDraws", "BlueDraws",
    "RedCurrentWinStreak", "BlueCurrentWinStreak", "RedCurrentLoseStreak", "BlueCurrentLoseStreak",
    "RedLongestWinStreak", "BlueLongestWinStreak",
    "RedTotalRoundsFought", "BlueTotalRoundsFought",
    "RedTotalTitleBouts", "BlueTotalTitleBouts",

    "RedWinsByDecisionUnanimous", "BlueWinsByDecisionUnanimous",
    "RedWinsByDecisionMajority", "BlueWinsByDecisionMajority",
    "RedWinsByDecisionSplit", "BlueWinsByDecisionSplit",
    "RedWinsByKO", "BlueWinsByKO",
    "RedWinsByTKODoctorStoppage", "BlueWinsByTKODoctorStoppage",
    "RedWinsBySubmission", "BlueWinsBySubmission",

    "RedAvgSigStrLanded", "BlueAvgSigStrLanded",
    "RedAvgSigStrPct", "BlueAvgSigStrPct",
    "RedAvgSubAtt", "BlueAvgSubAtt",
    "RedAvgTDLanded", "BlueAvgTDLanded",
    "RedAvgTDPct", "BlueAvgTDPct",

    "RedStance", "BlueStance",
    "RedAge", "BlueAge",
    "RedHeightCms", "BlueHeightCms",
    "RedReachCms", "BlueReachCms",
    "RedWeightLbs", "BlueWeightLbs",

    # Fighter rankings
    "RMatchWCRank", "BMatchWCRank", "RPFPRank", "BPFPRank", "RedRankStrength", "BlueRankStrength",

    # Differentials
    "WinDif", "LossDif", "WinStreakDif", "LoseStreakDif", "LongestWinStreakDif", "TotalRoundDif", "TotalTitleBoutDif",
    "DecisionDif", "KODif", "SigStrDif", "SigStrPctDif", "SubDif", "SubAttDif", "TDDif", "TDPctDif",
    "AgeDif", "HeightDif", "ReachDif",

    # Odds
    "RedOdds", "BlueOdds", "RedProb", "BlueProb", "RedExpectedValue", "BlueExpectedValue",
    "RedDecOdds", "BlueDecOdds", "RSubOdds", "BSubOdds", "RKOOdds", "BKOOdds",

    # Fight outcomes
    "Winner", "Finish", "FinishDetails", "FinishRound", "TotalFightTimeSecs"
]

# Helper functions for build_features()
def transform_rank(x):
    """
    Transform raw ranking values into a normalized strength score.

    Rules:
    - Missing value → 0 (unranked)
    - 0 → 16 (champion)
    - 1–15 → mapped to 15–1 (so higher = stronger)
    
    This ensures that larger values consistently represent stronger rankings.
    """
    if pd.isna(x):
        return 0
    elif x == 0:
        return 16
    else:
        return 16 - x

def odds_to_prob(odds):
    """
    Convert American odds to implied probability.

    - Negative odds: |odds| / (|odds| + 100)
    - Positive odds: 100 / (odds + 100)
    """
    odds = np.asarray(odds, dtype=float)
    return np.where(odds < 0, np.abs(odds) / (np.abs(odds) + 100), 100 / (odds + 100))

# Feature engineering function
def build_features(df):
    """
    Clean and preprocess raw UFC fight data.
    Applies feature engineering, transformations, and reordering.
    Returns a cleaned dataframe.
    """

    # Reset index & add FightID
    df = df.reset_index(drop=True)
    df["FightID"] = df.index + 1

    # Convert fight date to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Drop unused columns
    df = df.drop(columns=["Location", "EmptyArena", "FinishRoundTime"], errors="ignore")

    # Clean string columns
    df["Country"] = df["Country"].str.strip()
    df["Gender"] = df["Gender"].str.strip().str.capitalize()
    df["WeightClass"] = df["WeightClass"].str.replace("Catch Weight", "Catchweight")
    df["BlueStance"] = df["BlueStance"].str.strip()

    # Flip sign of differential columns so they represent Red − Blue
    dif_cols = [col for col in df.columns if col.endswith("Dif")]
    df[dif_cols] = df[dif_cols] * -1

    # Year
    df["Year"] = df["Date"].dt.year

    # Fighter-level history
    red_history = df[["RedFighter", "Date"]].rename(columns={"RedFighter": "Fighter"})
    blue_history = df[["BlueFighter", "Date"]].rename(columns={"BlueFighter": "Fighter"})
    fighter_history = pd.concat([red_history, blue_history], ignore_index=True)

    fighter_history = (
        fighter_history
        .sort_values(["Fighter", "Date"])
        .assign(DaysSinceLastFight=lambda f: f.groupby("Fighter")["Date"].diff().dt.days)
    )

    for color in ["Red", "Blue"]:
        df = df.merge(
            fighter_history.rename(
                columns={
                    "Fighter": f"{color}Fighter",
                    "DaysSinceLastFight": f"{color}DaysSinceLastFight"
                }
            ),
            on=[f"{color}Fighter", "Date"],
            how="left"
        )

    # Debut flags
    df["IsRedDebut"] = df["RedDaysSinceLastFight"].isna().astype(int)
    df["IsBlueDebut"] = df["BlueDaysSinceLastFight"].isna().astype(int)

    # Ranking transformations
    rank_cols = [c for c in df.columns if c.endswith("Rank") and c != "BetterRank"]
    df[rank_cols] = df[rank_cols].applymap(transform_rank)

    df["MatchWCRankDif"] = df["RMatchWCRank"] - df["BMatchWCRank"]
    df["PFPRankDif"] = df["RPFPRank"] - df["BPFPRank"]

    rank_cols_red = [c for c in rank_cols if c.startswith("R") and c not in ["RMatchWCRank", "RPFPRank"]]
    rank_cols_blue = [c for c in rank_cols if c.startswith("B") and c not in ["BMatchWCRank", "BPFPRank"]]

    df["RedRankStrength"] = df[rank_cols_red].max(axis=1)
    df["BlueRankStrength"] = df[rank_cols_blue].max(axis=1)
    df["RankStrengthDif"] = df["RedRankStrength"] - df["BlueRankStrength"]

    df = df.drop(columns=rank_cols_red + rank_cols_blue + ["BetterRank"], errors="ignore")

    # Combine Red/Blue decision totals
    df["RedWinsByDecision"] = (
        df["RedWinsByDecisionMajority"]
        + df["RedWinsByDecisionSplit"]
        + df["RedWinsByDecisionUnanimous"]
    )
    df["BlueWinsByDecision"] = (
        df["BlueWinsByDecisionMajority"]
        + df["BlueWinsByDecisionSplit"]
        + df["BlueWinsByDecisionUnanimous"]
    )

    # Create additional differentials
    diff_stats = ["WinsByDecision", "AvgSigStrPct", "AvgTDPct"]
    for stat in diff_stats:
        df[f"{stat}Dif"] = df[f"Red{stat}"] - df[f"Blue{stat}"]

    df = df.rename(columns={
        "WinsByDecisionDif": "DecisionDif",
        "AvgSigStrPctDif": "SigStrPctDif",
        "AvgSubAttDif": "SubAttDif",
        "AvgTDDif": "TDDif",
        "AvgTDPctDif": "TDPctDif"
    })

    df = df.drop(columns=["RedWinsByDecision", "BlueWinsByDecision"], errors="ignore")

    # Odds → implied probability
    df["RedProb"] = odds_to_prob(df["RedOdds"])
    df["BlueProb"] = odds_to_prob(df["BlueOdds"])

    # Reorder dataset
    df = df[COL_ORDER]

    return df

# --------------------
# Define Prediction Function
# --------------------

def predict_fight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict win probabilities for UFC fights.

    Takes a raw fight-level DataFrame, applies the same feature engineering,
    preprocessing, and PCA used during training, and returns predicted win
    probabilities for each possible outcome.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per fight and one column per class
        (e.g., Red vs Blue), containing predicted probabilities that
        sum to 1 across each row.
    """

    # Keep identifier
    fight_id = df["FightID"].reset_index(drop=True)

    # 1. Build features
    X = build_features(df)

    # 2. Preprocess
    X_pca = preprocess.transform(X)

    # 3. Predict
    probs = model.predict_proba(X_pca)

    # 4. Build output
    out = pd.DataFrame(
        probs,
        columns=model.classes_
    )

    out.insert(0, "FightID", fight_id)

    return out