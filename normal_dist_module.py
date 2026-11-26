import io
import math
import random
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import streamlit as st

import ai_marking_component


# --- Exact Normal Distribution Math Helpers (No scipy required) ---
def norm_cdf(z):
    """Cumulative distribution function for standard normal N(0, 1)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, accurate to ~1e-9)."""
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )


def norm_pdf(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


# --- Problem Generator ---
QUESTION_TYPES = [
    "z_score",
    "prob_above",
    "prob_below",
    "prob_between",
    "inv_x",
    "find_param",
    "expected_val",
]


def _pick_question_type(focus="All"):
    if focus == "Z-Scores":
        return "z_score"
    if focus == "Probabilities (%)":
        return random.choice(["prob_above", "prob_below", "prob_between"])
    if focus == "Inverse Normal (Find X)":
        return "inv_x"
    if focus == "Mean & Std Dev":
        return "find_param"
    if focus == "Expected Value":
        return "expected_val"
    return random.choice(QUESTION_TYPES)


def generate_normal_problem(focus="All", forced_qtype=None):
    qtype = forced_qtype if forced_qtype else _pick_question_type(focus)

    mu = random.choice([20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200])
    sigma = random.choice([2, 3,