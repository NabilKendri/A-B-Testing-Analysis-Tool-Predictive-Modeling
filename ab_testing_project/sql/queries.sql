-- Analysis queries against the `players` table.
-- Each query answers one specific business question about the A/B test.

-- 1. Sample size and retention rate per group.
-- (Same numbers the Python z-test uses -- SQL is the source of truth here.)
-- Median is computed with window functions: rank each player's rounds
-- within their group, then average the middle one or two ranks.
WITH ordered AS (
    SELECT
        version,
        sum_gamerounds,
        ROW_NUMBER() OVER (PARTITION BY version ORDER BY sum_gamerounds) AS rn,
        COUNT(*)     OVER (PARTITION BY version)                        AS cnt
    FROM players
),
medians AS (
    SELECT version, AVG(sum_gamerounds) AS median_rounds
    FROM ordered
    WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
    GROUP BY version
)
SELECT
    p.version,
    COUNT(*)                             AS n_players,
    ROUND(AVG(p.retention_1) * 100, 2)   AS retention_1_pct,
    ROUND(AVG(p.retention_7) * 100, 2)   AS retention_7_pct,
    ROUND(AVG(p.sum_gamerounds), 1)      AS avg_rounds,
    m.median_rounds
FROM players AS p
JOIN medians AS m ON m.version = p.version
GROUP BY p.version, m.median_rounds;


-- 2. Engagement segmentation: bucket players by how much they played,
-- then compare retention across buckets AND versions. This is the kind
-- of "what segment is driving the effect" question a PM asks after
-- seeing the headline A/B result.
SELECT
    version,
    CASE
        WHEN sum_gamerounds = 0        THEN '0 rounds (never started)'
        WHEN sum_gamerounds BETWEEN 1 AND 10   THEN '1-10 rounds (light)'
        WHEN sum_gamerounds BETWEEN 11 AND 50  THEN '11-50 rounds (moderate)'
        WHEN sum_gamerounds BETWEEN 51 AND 200 THEN '51-200 rounds (heavy)'
        ELSE '200+ rounds (whale)'
    END                                     AS engagement_bucket,
    COUNT(*)                               AS n_players,
    ROUND(AVG(retention_7) * 100, 2)       AS retention_7_pct
FROM players
GROUP BY version, engagement_bucket
ORDER BY
    version,
    MIN(sum_gamerounds);


-- 3. Activation check: of players who quit almost immediately (0 rounds
-- played), does gate version affect whether they even try the game?
-- If the gate change hurt activation itself, that would show up here,
-- separate from the retention effect on players who did engage.
SELECT
    version,
    ROUND(100.0 * SUM(CASE WHEN sum_gamerounds = 0 THEN 1 ELSE 0 END) / COUNT(*), 2)
        AS pct_never_played_a_round
FROM players
GROUP BY version;


-- 4. Window function example: within each version, rank players by
-- engagement and check retention for the top decile ("power users")
-- vs. everyone else -- do the heaviest players retain the same
-- regardless of gate version?
WITH ranked AS (
    SELECT
        *,
        NTILE(10) OVER (PARTITION BY version ORDER BY sum_gamerounds) AS decile
    FROM players
)
SELECT
    version,
    CASE WHEN decile = 10 THEN 'top 10% (power users)' ELSE 'bottom 90%' END AS segment,
    COUNT(*)                            AS n_players,
    ROUND(AVG(retention_7) * 100, 2)    AS retention_7_pct
FROM ranked
GROUP BY version, segment
ORDER BY version, segment;
