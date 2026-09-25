-- Smart City Traffic Intelligence - Part 1 SQL Analysis
-- Dataset table name: traffic
-- SQLite-compatible queries used in Part 1.

-- ============================================================
-- Task 1.1 - Verify dataset load
-- ============================================================
SELECT COUNT(*) AS row_count
FROM traffic;

SELECT *
FROM traffic
LIMIT 10;

-- ============================================================
-- Task 1.2 - Annual traffic trends, 2012-2017
-- ============================================================
SELECT
    strftime('%Y', date_time) AS year,
    SUM(traffic_volume) AS total_traffic_volume
FROM traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY strftime('%Y', date_time)
ORDER BY year;

-- Record coverage by year (used to qualify annual-total comparisons)
SELECT
    strftime('%Y', date_time) AS year,
    COUNT(*) AS records
FROM traffic
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- ============================================================
-- Task 1.3 - Temperature around New Year's Day and Labor Day
-- ============================================================
SELECT DISTINCT
    strftime('%Y', date_time) AS year,
    holiday,
    temp,
    traffic_volume
FROM traffic
WHERE holiday IN ('New Years Day', 'Labor Day')
  AND strftime('%Y', date_time) BETWEEN '2015' AND '2017'
ORDER BY year, holiday;

-- ============================================================
-- Task 2.1 - Descriptive traffic statistics
-- ============================================================
SELECT
    ROUND(AVG(traffic_volume), 2) AS mean,
    MIN(traffic_volume) AS minimum,
    MAX(traffic_volume) AS maximum,
    MAX(traffic_volume) - MIN(traffic_volume) AS range
FROM traffic;

-- Median for 48,204 records: average the two central ordered values.
SELECT AVG(traffic_volume) AS median
FROM (
    SELECT traffic_volume
    FROM traffic
    ORDER BY traffic_volume
    LIMIT 2
    OFFSET 24101
);

-- Population variance: E[X^2] - E[X]^2
SELECT
    ROUND(
        AVG(traffic_volume * traffic_volume)
        - AVG(traffic_volume) * AVG(traffic_volume),
        2
    ) AS variance
FROM traffic;

-- Note: the online SQLite environment used for this project did not
-- provide SQRT(), so standard deviation was calculated as sqrt(variance).

-- ============================================================
-- Task 3.1 - Basic probability counts
-- ============================================================
SELECT COUNT(*) AS congestion_records
FROM traffic
WHERE traffic_volume > 5500;

SELECT COUNT(*) AS clear_records
FROM traffic
WHERE weather_main = 'Clear';

SELECT COUNT(*) AS congestion_and_clear
FROM traffic
WHERE traffic_volume > 5500
  AND weather_main = 'Clear';

-- ============================================================
-- Task 3.2 - Conditional probability and independence inputs
-- ============================================================
SELECT
    COUNT(*) AS high_temp_congestion
FROM traffic
WHERE traffic_volume > 5500
  AND temp > 292;

-- P(Clear | Congestion)
SELECT 1763.0 / 7100 AS p_clear_given_congestion;

-- P(High Temperature | Congestion)
SELECT 1867.0 / 7100 AS p_high_temp_given_congestion;

-- Expected joint probability under independence:
-- P(Congestion) * P(Clear)
SELECT
    (7100.0 / 48204) * (13391.0 / 48204)
    AS expected_if_independent;

-- Clear vs Clouds counts used for the congestion odds ratio
SELECT
    weather_main,
    COUNT(*) AS total,
    SUM(traffic_volume > 5500) AS congested
FROM traffic
WHERE weather_main IN ('Clear', 'Clouds')
GROUP BY weather_main;

-- Odds ratio = clear congestion odds / cloudy congestion odds
SELECT
    ROUND((1763.0 / 11628) / (2592.0 / 12572), 3)
    AS odds_ratio;
