"""The shared methodology. Each module used to be copied into 2-5 sector repos; it now exists once.

decay       task-order decay weights and per-vacancy mean / variance of exposure
quadrants   the paper's four-category table (fixed and sector-quartile thresholds)
valuation   opportunity, cashable and non-cashable £ per unit of weight
rake        iterative proportional fitting to one-way margins, plus diagnostics
stats       weighted quantile
"""
