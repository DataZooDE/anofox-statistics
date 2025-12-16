import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automatic Identification of Demand (AID)

    The Failure of "Syntactic" Classification. The traditional approach to supply chain analysis, known as Syntactic Classification (SBC), relies on calculating summary statistics—specifically the Average Demand Interval (ADI) and the Coefficient of Variation squared ($CV^2$) — and mapping them to a matrix with fixed thresholds (classically $ADI=1.32$ and $CV^2=0.49$). The paper [Why do zeroes happen? A model-based approach for demand classification](https://arxiv.org/html/2504.05894v1) (Svetunkov & Sroginis 2025) identifies critical failures in this industry-standard approach:
    - **Arbitrary Boundaries:** The cutoffs (like 1.32) are mathematically derived from specific theoretical assumptions that rarely hold in real-world data. A product with an ADI of 1.31 is treated fundamentally differently from one with 1.33, causing "classification flip-flopping" where stable products jump between categories (e.g., from "Smooth" to "Lumpy") due to minor noise
    - **Ambiguity of Zeros**: SBC counts zeros but ignores their context. It cannot distinguish between Structural Zeros (product not yet launched), Censored Zeros (stockouts), and Stochastic Zeros (pure random intermittency).
    - **Loss of Distributional Information:** Knowing a product is "Erratic" is not enough to optimise inventory. You need to know if the demand follows a Gamma, Log-Normal, or Negative Binomial distribution to calculate safety stock accurately.

    This DuckDB extension implements the Model-Based Classification (MBC) approach proposed in the paper, which is also in the R package [greybox](https://cran.r-project.org/web/packages/greybox/index.html) available. Instead of relying on brittle ADI/CV thresholds, multiple probabilistic models (Poisson, Negative Binomial, Normal, etc.) are fitted to the data. It selects the classification based on Information Criteria (AIC), identifying the model that minimises information loss. This provides a rigorous, statistically stable foundation for all subsequent supply chain decisions.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        INSTALL anofox_statistics FROM community;
        LOAD anofox_statistics;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 1: Basic Demand Classification

    **Motivation**

    Standard classification rules are highly sensitive to outliers. A single extreme sales spike can inflate the standard deviation, pushing the $CV^2$ above 0.49 and reclassifying a stable product as "Erratic." This forces the use of complex forecasting models when a simple average would have sufficed.

    **Description**

    This query runs `aid_agg(demand)` to classify a series. Unlike standard SQL math that blindly checks variance thresholds, this function tests if the data is better explained by a stable process or a variable process, preventing "false erratic" classifications caused by single outliers.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        SELECT UNNEST(aid_agg(demand)) as classification
        FROM (VALUES (10), (12), (8), (15), (11), (9), (14), (10), (13), (11)) AS t(demand);
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 2: Intermittent Demand Detection

    **Motivation**

    In the Syntactic approach, intermittency is defined strictly by the average time between sales (ADI). However, on short time series, the ADI is unreliable. A slow-moving product might accidentally have two sales in consecutive weeks, lowering its ADI and tricking the system into treating it as a fast mover ("False Smooth"), leading to dangerous stockouts later.

    **Description**

    This example extracts `is_intermittent` by validating the process rather than just the interval. It checks if the sequence of zeros and non-zeros fits a Poisson-like arrival process, correctly identifying intermittency even when random clustering makes the demand look temporarily smooth.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        SELECT
            result.demand_type,
            result.is_intermittent,
            result.distribution,
            ROUND(result.zero_proportion, 2) AS zero_proportion,
            result.n_observations
        FROM (
            SELECT aid_agg(demand) AS result
            FROM (VALUES (0), (0), (5), (0), (8), (0), (3), (0), (0), (6)) AS t(demand)
        );
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 3: Multi-SKU Classification with GROUP BY

    **Motivation**

    Applying a "one-size-fits-all" threshold (e.g., "Intermittent if ADI > 1.32") across a diverse catalogue fails because different categories (e.g., Screws vs. Engines) have different natural variances. Syntactic rules force analysts to manually maintain complex lookup tables of "exception rules" for other item types.

    **Description**

    This query demonstrates how to scale the Model-Based approach. Because it uses Information Criteria (relative fit) rather than absolute thresholds, the engine automatically adapts to the scale and behaviour of every individual SKU, eliminating the need for manual category-specific tuning.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        WITH sales AS (
            -- SKU001: Intermittent (many zeros)
            SELECT 'SKU001' as sku, val as demand, row_number() OVER () as period
            FROM (VALUES (0), (0), (5), (0), (8), (0), (3), (0), (0), (6)) AS t(val)
            UNION ALL
            -- SKU002: Regular (no zeros, consistent)
            SELECT 'SKU002' as sku, val as demand, row_number() OVER () as period
            FROM (VALUES (45), (48), (42), (50), (47), (44), (49), (46), (51), (43)) AS t(val)
            UNION ALL
            -- SKU003: New product (leading zeros)
            SELECT 'SKU003' as sku, val as demand, row_number() OVER () as period
            FROM (VALUES (0), (0), (0), (0), (5), (8), (12), (15), (18), (20)) AS t(val)
            UNION ALL
            -- SKU004: Obsolete product (trailing zeros)
            SELECT 'SKU004' as sku, val as demand, row_number() OVER () as period
            FROM (VALUES (25), (22), (18), (15), (10), (5), (0), (0), (0), (0)) AS t(val)
        )
        SELECT
            sku,
            result.demand_type,
            result.distribution,
            ROUND(result.mean, 1) AS mean,
            ROUND(result.zero_proportion, 2) AS zero_pct,
            result.is_new_product,
            result.is_obsolete_product,
            result.has_stockouts
        FROM (
            SELECT sku, aid_agg(demand ORDER BY period) AS result
            FROM sales
            GROUP BY sku
        ) sub
        ORDER BY sku;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 4: Simple Anomaly Detection Per Observation

    **Motivation**

    Most ERP systems use "3-Sigma" (Standard Deviation) to detect outliers. This assumes a Normal (Bell Curve) distribution. The paper argues this is disastrous for supply chains because demand is often skewed (right-tailed). A 3-Sigma rule will flag valid high demand as an "anomaly," causing planners to ignore real growth signals. Outlier detection is use case specific. This may be just a rough estimation and need addiotional adjustment and/or more sophisticated methods.

    **Description**

    This example uses aid_anomaly_agg to detect outliers based on the fitted distribution. If the data follows a Poisson distribution, the engine asks, "Is this value unlikely for a Poisson process?" rather than "Is it 3 deviations from the mean?", significantly reducing false positive anomaly flags.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        WITH demand_series AS (
            SELECT row_number() OVER () as period, demand
            FROM (VALUES (0), (0), (5), (0), (8), (0), (0)) AS t(demand)
        ),
        anomalies AS (
            SELECT aid_anomaly_agg(demand ORDER BY period) AS anomaly_flags
            FROM demand_series
        )
        SELECT
            ds.period,
            ds.demand,
            f.stockout,
            f.new_product,
            f.obsolete_product,
            f.high_outlier,
            f.low_outlier
        FROM demand_series ds, anomalies a, LATERAL UNNEST(a.anomaly_flags) WITH ORDINALITY AS t(f, ord)
        WHERE ds.period = t.ord
        ORDER BY ds.period;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 5: Custom Threshold for Intermittent Classification

    **Motivation**

    While Model-Based Classification is statistically superior, organisational inertia often mandates adherence to legacy business rules (e.g., "Marketing defines 'Slow' as selling less than 50% of the time"). Pure statistical engines often fail to adopt because they cannot accommodate these operational constraints.

    **Description**

    This query demonstrates the flexibility to override the statistical engine. By passing `{'intermittent_threshold': 0.5}`, it forces the system to respect a domain-specific constraint, allowing the user to blend the robustness of the AID engine with fixed business logic where necessary.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        WITH demand AS (
            SELECT val as demand FROM (VALUES (0), (5), (0), (8), (10), (0), (12), (7), (0), (9)) AS t(val)
        )
        SELECT
            'Default (30%)' AS threshold_setting,
            result.demand_type,
            ROUND(result.zero_proportion, 2) AS zero_proportion
        FROM (SELECT aid_agg(demand) AS result FROM demand)
        UNION ALL
        SELECT
            'Custom (50%)' AS threshold_setting,
            result.demand_type,
            ROUND(result.zero_proportion, 2) AS zero_proportion
        FROM (SELECT aid_agg(demand, {{'intermittent_threshold': 0.5}}) AS result FROM demand);
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 6: Stockout Analysis

    **Motivation**

    A major blind spot in Syntactic Classification is Censoring. A zero record in the database is ambiguous: it could mean "no customer arrived" or "customer arrived but no stock was available." Treating stockout zeros as demand zeros distorts the demand probability, leading to a downward-biased forecast that perpetuates the shortage.

    **Description**

    This query distinguishes "natural zeros" from "censored zeros" (stockouts). It implements the paper's requirement to account for inventory availability, ensuring that the demand profile reflects potential sales rather than just recorded sales.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        WITH inventory AS (
            -- Product A: Has stockouts (zeros in middle of positive demand)
            SELECT 'Product_A' as product, val as demand, row_number() OVER () as week
            FROM (VALUES (50), (45), (0), (0), (52), (48), (0), (55), (47), (51)) AS t(val)
            UNION ALL
            -- Product B: No stockouts (continuous positive demand)
            SELECT 'Product_B' as product, val as demand, row_number() OVER () as week
            FROM (VALUES (30), (32), (28), (35), (31), (29), (33), (30), (34), (32)) AS t(val)
            UNION ALL
            -- Product C: Many stockouts
            SELECT 'Product_C' as product, val as demand, row_number() OVER () as week
            FROM (VALUES (20), (0), (18), (0), (0), (22), (0), (19), (0), (21)) AS t(val)
        )
        SELECT
            product,
            result.has_stockouts,
            result.stockout_count,
            ROUND(result.stockout_count::DOUBLE / result.n_observations * 100, 1) AS stockout_pct,
            result.demand_type
        FROM (
            SELECT product, aid_agg(demand ORDER BY week) AS result
            FROM inventory
            GROUP BY product
        ) sub
        ORDER BY result.stockout_count DESC;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 7: Distribution Recommendation

    **Motivation**

    Knowing a product is "Lumpy" (Quadrant 4 in SBC) is descriptive but not actionable. To set a safety stock level that achieves a 95% service level, you need the specific Probability Mass Function (PMF). Syntactic rules do not provide this; they only categorize.

    **Description**

    This example outputs the recommended_distribution. It bridges the gap between classification and calculation, automating the selection of the mathematical model (e.g., Negative Binomial for overdispersed data) required for precise inventory optimization.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        WITH products AS (
            -- Count data (integers, low values) -> Poisson family
            SELECT 'Low_Count_Data' as category, val as demand
            FROM (VALUES (2), (3), (1), (4), (2), (3), (2), (5), (3), (2)) AS t(val)
            UNION ALL
            -- Overdispersed count data -> Negative Binomial
            SELECT 'Overdispersed_Counts' as category, val as demand
            FROM (VALUES (0), (0), (5), (0), (12), (0), (0), (8), (0), (15)) AS t(val)
            UNION ALL
            -- Continuous positive data -> Gamma/Lognormal
            SELECT 'Continuous_Positive' as category, val as demand
            FROM (VALUES (10.5), (12.3), (8.7), (15.2), (11.8), (9.4), (14.1), (10.9), (13.5), (11.2)) AS t(val)
            UNION ALL
            -- Normal-like data -> Normal
            SELECT 'Normal_Like' as category, val as demand
            FROM (VALUES (100), (102), (98), (101), (99), (103), (97), (100), (101), (99)) AS t(val)
        )
        SELECT
            category,
            result.distribution AS recommended_distribution,
            result.demand_type,
            ROUND(result.mean, 2) AS mean,
            ROUND(result.variance, 2) AS variance
        FROM (
            SELECT category, aid_agg(demand) AS result
            FROM products
            GROUP BY category
        ) sub
        ORDER BY category;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 8: Product Lifecycle Detection

    **Motivation**

    Syntactic metrics (ADI/CV) are static averages calculated over a fixed window (e.g., last 12 months). They fail to detect **Structural Breaks** — such as a product dying (End of Life) or being born (NPI). Including "dead" periods in the average creates a "Zombie" forecast that suggests buying stock for a product that will never sell again.

    **Description**

    This query identifies "Leading Zeros" and "Trailing Zeros." It filters out these non-demand periods to prevent them from contaminating the statistical properties of the active lifecycle, ensuring the classification reflects the product's current reality rather than its history.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        WITH lifecycle AS (
            SELECT 'NewProduct_2024' as product, val as demand, row_number() OVER () as month
            FROM (VALUES (0), (0), (0), (5), (12), (25), (40), (55), (70), (85), (95), (100)) AS t(val)
            UNION ALL
            SELECT 'MatureProduct' as product, val as demand, row_number() OVER () as month
            FROM (VALUES (80), (82), (78), (85), (81), (79), (83), (80), (84), (82), (81), (80)) AS t(val)
            UNION ALL
            SELECT 'EndOfLife_Legacy' as product, val as demand, row_number() OVER () as month
            FROM (VALUES (50), (42), (35), (28), (20), (12), (5), (0), (0), (0), (0), (0)) AS t(val)
        )
        SELECT
            product,
            CASE
                WHEN result.is_new_product AND NOT result.is_obsolete_product THEN 'Introduction Phase'
                WHEN result.is_obsolete_product AND NOT result.is_new_product THEN 'Decline Phase'
                WHEN result.is_new_product AND result.is_obsolete_product THEN 'Short Lifecycle'
                ELSE 'Mature/Stable'
            END AS lifecycle_stage,
            result.new_product_count AS intro_periods,
            result.obsolete_product_count AS decline_periods,
            result.n_observations AS total_periods
        FROM (
            SELECT product, aid_agg(demand ORDER BY month) AS result
            FROM lifecycle
            GROUP BY product
        ) sub
        ORDER BY product;
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example 10: Demand Analysis Report

    **Motivation**

    In reality, demand data suffers from all these issues simultaneously: non-normality, stockouts, lifecycle changes, and outliers. Disjointed tools that fix only one problem (e.g., just outlier detection) often make the others worse. The paper concludes that a robust system must model the joint probability of these events.

    **Description**

    This final example aggregates all AID features into a "Control Tower" view. It represents the Model-Based ideal: a holistic assessment that simultaneously cleans anomalies, accounts for stockouts, fits the distribution, and determines lifecycle status, delivering a single, scientifically valid truth for every SKU.
    """)
    return


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        CREATE OR REPLACE TABLE demand_data AS
        SELECT
            'SKU' || LPAD(CAST(sku_id AS VARCHAR), 3, '0') AS sku,
            CASE
                WHEN sku_id <= 3 THEN 'Electronics'
                WHEN sku_id <= 6 THEN 'Apparel'
                ELSE 'Food'
            END AS category,
            week,
            -- Generate different demand patterns
            CASE
                WHEN sku_id = 1 THEN GREATEST(0, 50 + CAST(FLOOR(RANDOM() * 20 - 10) AS INTEGER))
                WHEN sku_id = 2 THEN CASE WHEN RANDOM() < 0.4 THEN 0 ELSE CAST(FLOOR(RANDOM() * 30 + 10) AS INTEGER) END
                WHEN sku_id = 3 THEN CASE WHEN week <= 4 THEN 0 ELSE CAST(FLOOR(week * 5 + RANDOM() * 10) AS INTEGER) END
                WHEN sku_id = 4 THEN GREATEST(0, 100 + CAST(FLOOR(RANDOM() * 30 - 15) AS INTEGER))
                WHEN sku_id = 5 THEN CASE WHEN RANDOM() < 0.6 THEN 0 ELSE CAST(FLOOR(RANDOM() * 20 + 5) AS INTEGER) END
                WHEN sku_id = 6 THEN CASE WHEN week >= 9 THEN 0 ELSE GREATEST(0, 80 - week * 8 + CAST(FLOOR(RANDOM() * 10) AS INTEGER)) END
                WHEN sku_id = 7 THEN GREATEST(0, 200 + CAST(FLOOR(RANDOM() * 40 - 20) AS INTEGER))
                WHEN sku_id = 8 THEN CASE WHEN RANDOM() < 0.3 THEN 0 ELSE CAST(FLOOR(RANDOM() * 50 + 20) AS INTEGER) END
                ELSE GREATEST(0, 150 + CAST(FLOOR(RANDOM() * 50 - 25) AS INTEGER))
            END AS demand
        FROM (SELECT * FROM range(1, 10) t1(sku_id), range(1, 13) t2(week));

        SELECT
            sub.category,
            sub.sku,
            result.demand_type,
            result.distribution,
            ROUND(result.mean, 1) AS avg_demand,
            ROUND(result.zero_proportion * 100, 0) AS zero_pct,
            result.has_stockouts,
            result.stockout_count,
            CASE
                WHEN result.is_new_product THEN 'New'
                WHEN result.is_obsolete_product THEN 'EOL'
                ELSE 'Active'
            END AS status,
            result.high_outlier_count + result.low_outlier_count AS anomalies
        FROM (
            SELECT sku, category, aid_agg(demand ORDER BY week) AS result
            FROM demand_data
            GROUP BY sku, category
        ) sub
        ORDER BY sub.category, sub.sku;

        -- Summary by category
        SELECT
            category,
            COUNT(*) AS sku_count,
            SUM(CASE WHEN result.is_intermittent THEN 1 ELSE 0 END) AS intermittent_skus,
            SUM(CASE WHEN result.has_stockouts THEN 1 ELSE 0 END) AS skus_with_stockouts,
            SUM(CASE WHEN result.is_new_product THEN 1 ELSE 0 END) AS new_products,
            SUM(CASE WHEN result.is_obsolete_product THEN 1 ELSE 0 END) AS eol_products
        FROM (
            SELECT category, aid_agg(demand ORDER BY week) AS result
            FROM demand_data
            GROUP BY sku, category
        ) sub
        GROUP BY category
        ORDER BY category;
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
