-- Schema for the cleaned Cookie Cats A/B test data.
-- One row per player; loaded from data/cookie_cats_clean.csv.

DROP TABLE IF EXISTS players;

CREATE TABLE players (
    userid          INTEGER PRIMARY KEY,
    version         TEXT NOT NULL CHECK (version IN ('gate_30', 'gate_40')),
    sum_gamerounds  INTEGER NOT NULL,
    retention_1     INTEGER NOT NULL CHECK (retention_1 IN (0, 1)),
    retention_7     INTEGER NOT NULL CHECK (retention_7 IN (0, 1))
);

CREATE INDEX idx_players_version ON players(version);
