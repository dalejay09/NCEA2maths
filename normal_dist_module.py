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
        3.224671290700398e-01,
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

    # Choose clean, realistic numeric parameters
    mu = random.choice([20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150, 200])
    sigma = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15])
    while sigma * 3 >= mu:
        sigma = random.choice([2, 3, 4, 5])

    num_bins = random.randint(8, 20)
    # Generate sample histogram bar heights around the theoretical curve
    x_min = mu - 3.2 * sigma
    x_max = mu + 3.2 * sigma
    bin_width = (x_max - x_min) / num_bins
    sample_n = random.choice([200, 250, 300, 400, 500])

    bin_edges = [x_min + i * bin_width for i in range(num_bins + 1)]
    bin_counts = []
    for i in range(num_bins):
        mid = 0.5 * (bin_edges[i] + bin_edges[i + 1])
        expected_freq = sample_n * norm_pdf(mid, mu, sigma) * bin_width
        jitter = random.uniform(0.82, 1.18)
        bin_counts.append(max(1, round(expected_freq * jitter)))

    # Pick non-zero Z offsets so students work with genuine values
    z_candidates = [round(z * 0.1, 1) for z in range(-24, 25) if abs(z) >= 3]

    if qtype == "z_score":
        z_true = random.choice(z_candidates)
        x_val = round(mu + z_true * sigma, 2)
        if x_val.is_integer():
            x_val = int(x_val)
        z_ans = round((x_val - mu) / sigma, 3)

        prompt = (
            f"Mean (μ) = {mu}, Standard Deviation (σ) = {sigma}\n\n"
            f"Calculate the Z-score for X = {x_val}."
        )
        ans_str = f"Z = {z_ans:.2f}"
        rule = (
            f"Formula: Z = (X - μ) / σ = ({x_val} - {mu}) / {sigma} = {z_ans:.3f}. "
            f"Accept Z = {z_ans:.2f} (or {z_ans:.3f}). Sign (+ or -) must be correct."
        )
        highlight = {"type": "point", "x": x_val}

    elif qtype == "prob_above":
        z_true = random.choice(z_candidates)
        x_val = round(mu + z_true * sigma, 2)
        if x_val.is_integer():
            x_val = int(x_val)
        z_exact = (x_val - mu) / sigma
        pct = (1.0 - norm_cdf(z_exact)) * 100.0

        prompt = (
            f"Mean (μ) = {mu}, Standard Deviation (σ) = {sigma}\n\n"
            f"Calculate the percentage of the population above X = {x_val}."
        )
        ans_str = f"{pct:.2f}% (or {pct:.1f}%)"
        rule = (
            f"P(X > {x_val}) = P(Z > {z_exact:.2f}) = {pct:.2f}%. "
            f"Accept answers rounded to 1 or 2 decimal places (approx {pct:.1f}% to {pct:.2f}%), "
            f"or the equivalent decimal probability ({pct/100:.4f}) if clearly converted/stated. "
            f"Bonus if the student shades the right tail on the diagram."
        )
        highlight = {"type": "above", "x": x_val}

    elif qtype == "prob_below":
        z_true = random.choice(z_candidates)
        x_val = round(mu + z_true * sigma, 2)
        if x_val.is_integer():
            x_val = int(x_val)
        z_exact = (x_val - mu) / sigma
        pct = norm_cdf(z_exact) * 100.0

        prompt = (
            f"Mean (μ) = {mu}, Standard Deviation (σ) = {sigma}\n\n"
            f"Calculate the percentage of the population below X = {x_val}."
        )
        ans_str = f"{pct:.2f}% (or {pct:.1f}%)"
        rule = (
            f"P(X < {x_val}) = P(Z < {z_exact:.2f}) = {pct:.2f}%. "
            f"Accept answers rounded to 1 or 2 decimal places (approx {pct:.1f}% to {pct:.2f}%), "
            f"or equivalent decimal probability ({pct/100:.4f}). "
            f"Bonus if the student shades the left tail on the diagram."
        )
        highlight = {"type": "below", "x": x_val}

    elif qtype == "prob_between":
        z1 = random.choice([round(z * 0.1, 1) for z in range(-22, -1)])
        z2 = random.choice([round(z * 0.1, 1) for z in range(2, 23)])
        x1 = round(mu + z1 * sigma, 2)
        x2 = round(mu + z2 * sigma, 2)
        if x1.is_integer():
            x1 = int(x1)
        if x2.is_integer():
            x2 = int(x2)
        pct = (norm_cdf((x2 - mu) / sigma) - norm_cdf((x1 - mu) / sigma)) * 100.0

        prompt = (
            f"Mean (μ) = {mu}, Standard Deviation (σ) = {sigma}\n\n"
            f"Calculate the percentage of the population between X = {x1} and X = {x2}."
        )
        ans_str = f"{pct:.2f}% (or {pct:.1f}%)"
        rule = (
            f"P({x1} < X < {x2}) = {pct:.2f}%. "
            f"Accept answers rounded to 1 or 2 decimal places (approx {pct:.1f}% to {pct:.2f}%). "
            f"Bonus if the student shades the central region between {x1} and {x2}."
        )
        highlight = {"type": "between", "x1": x1, "x2": x2}

    elif qtype == "inv_x":
        direction = random.choice(["above", "below"])
        target_pct = random.choice([2, 5, 10, 12, 15, 20, 25, 30, 35, 40, 60, 70, 75, 80, 85, 90, 95])
        cum_p = (1.0 - target_pct / 100.0) if direction == "above" else (target_pct / 100.0)
        z_req = norm_ppf(cum_p)
        x_ans = mu + z_req * sigma

        prompt = (
            f"Mean (μ) = {mu}, Standard Deviation (σ) = {sigma}\n\n"
            f"Find the value of X if {target_pct}% of the population lies {direction} X."
        )
        ans_str = f"X = {x_ans:.2f} (or {x_ans:.1f})"
        rule = (
            f"Cumulative area to the left = {cum_p:.2f}, giving Z = {z_req:.3f}. "
            f"X = {mu} + ({z_req:.3f} × {sigma}) = {x_ans:.2f}. "
            f"Accept X rounded to 1 or 2 decimal places (within ±0.15 of {x_ans:.2f})."
        )
        highlight = {"type": "none"}

    elif qtype == "find_param":
        missing = random.choice(["mean", "sd"])
        direction = random.choice(["above", "below"])
        target_pct = random.choice([5, 10, 15, 20, 25, 30, 75, 80, 85, 90])
        cum_p = (1.0 - target_pct / 100.0) if direction == "above" else (target_pct / 100.0)
        z_req = norm_ppf(cum_p)

        # Ensure z_req is not close to 0
        if abs(z_req) < 0.25:
            target_pct = 15
            cum_p = 0.85 if direction == "above" else 0.15
            z_req = norm_ppf(cum_p)

        x_val = round(mu + z_req * sigma, 1)

        if missing == "mean":
            mu_ans = x_val - z_req * sigma
            prompt = (
                f"Standard Deviation (σ) = {sigma} (Mean μ is unknown)\n\n"
                f"Given that {target_pct}% of the population lies {direction} X = {x_val}, "
                f"calculate the mean (μ)."
            )
            ans_str = f"μ = {mu_ans:.2f} (or {mu_ans:.1f})"
            rule = (
                f"Z = {z_req:.3f}. Using Z = (X - μ)/σ -> {z_req:.3f} = ({x_val} - μ)/{sigma}, "
                f"so μ = {mu_ans:.2f}. Accept within ±0.15 of {mu_ans:.2f}."
            )
            # Re-center histogram around true mu_ans for visual consistency
            x_min = mu_ans - 3.2 * sigma
            x_max = mu_ans + 3.2 * sigma
            bin_width = (x_max - x_min) / num_bins
            bin_edges = [x_min + i * bin_width for i in range(num_bins + 1)]
            mu_plot, sigma_plot = mu_ans, sigma
        else:
            sigma_ans = (x_val - mu) / z_req
            prompt = (
                f"Mean (μ) = {mu} (Standard Deviation σ is unknown)\n\n"
                f"Given that {target_pct}% of the population lies {direction} X = {x_val}, "
                f"calculate the standard deviation (σ)."
            )
            ans_str = f"σ = {sigma_ans:.2f} (or {sigma_ans:.1f})"
            rule = (
                f"Z = {z_req:.3f}. Using Z = (X - μ)/σ -> {z_req:.3f} = ({x_val} - {mu})/σ, "
                f"so σ = {sigma_ans:.2f}. Accept within ±0.15 of {sigma_ans:.2f}."
            )
            mu_plot, sigma_plot = mu, sigma_ans

        highlight = {"type": "none"}
        return {
            "qtype": qtype,
            "mode": "Written",
            "text": prompt,
            "ans": ans_str,
            "grading_rule": rule,
            "mu": mu_plot,
            "sigma": sigma_plot,
            "hide_mu": (missing == "mean"),
            "hide_sigma": (missing == "sd"),
            "bin_edges": bin_edges,
            "bin_counts": bin_counts,
            "sample_n": sample_n,
            "highlight": highlight,
        }

    else:  # expected_val
        n_pop = random.choice([120, 150, 200, 250, 300, 400, 500, 600, 800, 1000])
        sub_dir = random.choice(["above", "below", "between"])

        if sub_dir in ["above", "below"]:
            z_true = random.choice(z_candidates)
            x_val = round(mu + z_true * sigma, 2)
            if x_val.is_integer():
                x_val = int(x_val)
            z_exact = (x_val - mu) / sigma
            prob = (1.0 - norm_cdf(z_exact)) if sub_dir == "above" else norm_cdf(z_exact)
            exp_val = n_pop * prob
            cond_txt = f"{sub_dir} X = {x_val}"
            highlight = {"type": sub_dir, "x": x_val}
        else:
            z1 = random.choice([round(z * 0.1, 1) for z in range(-20, -2)])
            z2 = random.choice([round(z * 0.1, 1) for z in range(2, 21)])
            x1 = round(mu + z1 * sigma, 2)
            x2 = round(mu + z2 * sigma, 2)
            if x1.is_integer():
                x1 = int(x1)
            if x2.is_integer():
                x2 = int(x2)
            prob = norm_cdf((x2 - mu) / sigma) - norm_cdf((x1 - mu) / sigma)
            exp_val = n_pop * prob
            cond_txt = f"between X = {x1} and X = {x2}"
            highlight = {"type": "between", "x1": x1, "x2": x2}

        prompt = (
            f"Mean (μ) = {mu}, Standard Deviation (σ) = {sigma}\n\n"
            f"In a sample of n = {n_pop}, calculate the expected number of values {cond_txt}."
        )
        ans_str = f"{exp_val:.1f} (or {round(exp_val)})"
        rule = (
            f"Probability p = {prob:.4f}. Expected value E = n × p = {n_pop} × {prob:.4f} = {exp_val:.2f}. "
            f"Accept exact decimal ({exp_val:.1f} or {exp_val:.2f}) or rounded whole number ({round(exp_val)})."
        )

    return {
        "qtype": qtype,
        "mode": "Written",
        "text": prompt,
        "ans": ans_str,
        "grading_rule": rule,
        "mu": mu,
        "sigma": sigma,
        "hide_mu": False,
        "hide_sigma": False,
        "bin_edges": bin_edges,
        "bin_counts": bin_counts,
        "sample_n": sample_n,
        "highlight": highlight,
    }


# --- Visual Engine: Histogram + Optional Normal Curve ---
def draw_normal_problem_image(p_data, show_curve=False, width_px=420, height_px=480):
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 2.35], hspace=0.18)

    # Top panel: Question text
    ax_txt = fig.add_subplot(gs[0])
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")

    wrapped_lines = []
    for line in p_data["text"].split("\n"):
        if line.strip() == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=44))
    ax_txt.text(
        0.02,
        0.96,
        "\n".join(wrapped_lines),
        fontsize=11.5,
        ha="left",
        va="top",
        family="sans-serif",
        color="#111111",
    )

    # Bottom panel: Histogram (8-20 bars)
    ax = fig.add_subplot(gs[1])
    edges = p_data["bin_edges"]
    counts = p_data["bin_counts"]
    widths = [edges[i + 1] - edges[i] for i in range(len(counts))]

    ax.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color="#A7C7E7",
        edgecolor="#2B547E",
        linewidth=1.0,
        alpha=0.85,
        zorder=2,
    )

    mu = p_data["mu"]
    sigma = p_data["sigma"]
    bin_w = widths[0]
    total_count = sum(counts)

    # Optional smooth Normal curve overlay
    if show_curve:
        x_vals = [
            edges[0] + (edges[-1] - edges[0]) * k / 200.0 for k in range(201)
        ]
        y_vals = [norm_pdf(x, mu, sigma) * total_count * bin_w for x in x_vals]
        ax.plot(
            x_vals,
            y_vals,
            color="#D92B2B",
            linestyle="--",
            linewidth=2.0,
            label="Normal Curve",
            zorder=4,
        )

    # Subtle mean reference line if mean is known
    if not p_data.get("hide_mu", False):
        ax.axvline(mu, color="#333333", linestyle=":", linewidth=1.2, zorder=3)

    # Summary badge in upper-right corner of plot
    mu_label = "?" if p_data.get("hide_mu", False) else f"{mu}"
    sd_label = "?" if p_data.get("hide_sigma", False) else f"{sigma}"
    ax.text(
        0.97,
        0.93,
        f"μ = {mu_label}\nσ = {sd_label}\nbins = {len(counts)}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        zorder=5,
    )

    ax.set_xlabel("X", fontsize=10, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=10)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.13, right=0.96, top=0.96, bottom=0.11)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", transparent=False)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA").copy()


# --- Worksheet PDF Generator ---
def create_nd_pdf_bytes(focus="All"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        problems = [generate_normal_problem(focus=focus) for _ in range(12)]

        # Page 1: 6 Questions with histograms (2x3 grid)
        for page_start in [0, 6]:
            fig_ws, axes = plt.subplots(3, 2, figsize=(8.27, 11.69))
            fig_ws.subplots_adjust(
                left=0.08, right=0.95, top=0.91, bottom=0.05, wspace=0.25, hspace=0.45
            )
            fig_ws.suptitle(
                "NCEA Level 2 Maths: Normal Distributions Worksheet",
                fontsize=14,
                fontweight="bold",
            )

            for idx in range(6):
                q_idx = page_start + idx
                p = problems[q_idx]
                row, col = divmod(idx, 2)
                ax = axes[row, col]

                edges = p["bin_edges"]
                counts = p["bin_counts"]
                widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
                ax.bar(
                    edges[:-1],
                    counts,
                    width=widths,
                    align="edge",
                    color="#A7C7E7",
                    edgecolor="#2B547E",
                    linewidth=0.7,
                )
                title_txt = f"Q{q_idx + 1}: " + p["text"].replace("\n\n", " | ")
                ax.set_title("\n".join(textwrap.wrap(title_txt, width=42)), fontsize=7.5, loc="left")
                ax.tick_params(labelsize=6.5)

            pdf.savefig(fig_ws)
            plt.close(fig_ws)

        # Page 3: Answer Key
        fig_ans, ax_ans = plt.subplots(figsize=(8.27, 11.69))
        ax_ans.axis("off")
        ax_ans.text(0.5, 0.95, "Answer Key", fontsize=15, fontweight="bold", ha="center")
        for i, p in enumerate(problems):
            y_pos = 0.88 - i * 0.065
            ax_ans.text(0.08, y_pos, f"Q{i + 1}: {p['ans']}", fontsize=10, va="top")
        pdf.savefig(fig_ans)
        plt.close(fig_ans)

    buffer.seek(0)
    return buffer.getvalue()


# --- Queue & Settings Handlers ---
def handle_nd_settings_change():
    st.session_state.nd_generating = True
    st.session_state.nd_pdf_bytes = None
    st.session_state.nd_suite_refresh_id += 1
    st.session_state.nd_question_queue = []
    st.session_state.nd_quiz_active = False
    st.session_state.nd_quiz_score = 0
    st.session_state.nd_questions_answered = 0
    st.session_state.nd_quiz_completed = False


def get_next_nd_question():
    mode = st.session_state.get("nd_selection_mode", "Random")
    focus = st.session_state.get("nd_focus", "All")

    if mode == "Random":
        return generate_normal_problem(focus=focus)

    if "nd_question_queue" not in st.session_state or not st.session_state.nd_question_queue:
        if mode == "Quiz":
            st.session_state.nd_quiz_score = 0
            st.session_state.nd_questions_answered = 0
            st.session_state.nd_quiz_completed = False

        if focus == "All":
            qtypes = list(QUESTION_TYPES)
        elif focus == "Probabilities (%)":
            qtypes = ["prob_above", "prob_below", "prob_between"]
        else:
            qtypes = [_pick_question_type(focus)] * 5

        random.shuffle(qtypes)
        if mode == "Quiz" and len(qtypes) > 10:
            qtypes = qtypes[:10]

        st.session_state.nd_question_queue = qtypes

    if mode == "Quiz" and not st.session_state.nd_question_queue:
        st.session_state.nd_quiz_completed = True
        return None

    next_qtype = st.session_state.nd_question_queue.pop(0)
    return generate_normal_problem(focus=focus, forced_qtype=next_qtype)


# --- START OF MODULE WRAPPER ---
def run_app():
    st.markdown(
        """
        <style>
        button[kind="primary"] { background-color: #007AFF !important; border-color: #007AFF !important; color: white !important; }
        button[kind="primary"]:hover { background-color: #0056b3 !important; border-color: #0056b3 !important; }
        div[data-testid="stElementContainer"]:has(#next-problem-btn) + div[data-testid="stElementContainer"] button {
            background-color: #28a745 !important; border-color: #28a745 !important; color: white !important;
        }
        div[data-testid="stElementContainer"]:has(#next-problem-btn) + div[data-testid="stElementContainer"] button:hover {
            background-color: #218838 !important; border-color: #218838 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize isolated nd_ state variables
    if "nd_generating" not in st.session_state:
        st.session_state.nd_generating = True
    if "nd_pdf_bytes" not in st.session_state:
        st.session_state.nd_pdf_bytes = None
    if "nd_suite_refresh_id" not in st.session_state:
        st.session_state.nd_suite_refresh_id = 0
    if "nd_selection_mode" not in st.session_state:
        st.session_state.nd_selection_mode = "Random"
    if "nd_focus" not in st.session_state:
        st.session_state.nd_focus = "All"
    if "nd_show_curve" not in st.session_state:
        st.session_state.nd_show_curve = False
    if "nd_quiz_active" not in st.session_state:
        st.session_state.nd_quiz_active = False
    if "nd_questions_answered" not in st.session_state:
        st.session_state.nd_questions_answered = 0
    if "nd_quiz_score" not in st.session_state:
        st.session_state.nd_quiz_score = 0
    if "nd_quiz_completed" not in st.session_state:
        st.session_state.nd_quiz_completed = False

    st.title("Normal Distributions 📊")

    col_actions, col_set = st.columns([5, 1])
    with col_actions:
        with st.popover("📄 Worksheet Actions", use_container_width=True):
            if st.session_state.nd_pdf_bytes is None:
                if st.button("⚙️ Generate 12-Question PDF", use_container_width=True, key="nd_gen_pdf"):
                    with st.spinner("Compiling PDF..."):
                        st.session_state.nd_pdf_bytes = create_nd_pdf_bytes(st.session_state.nd_focus)
                    st.rerun()
            else:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                st.download_button(
                    "⬇️ Download Worksheet",
                    data=st.session_state.nd_pdf_bytes,
                    file_name=f"Normal_Distributions_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary",
                    key="nd_dl_pdf",
                )

    with col_set:
        with st.popover("⚙️", use_container_width=True):
            st.radio(
                "Selection Mode",
                ["Random", "Cyclical", "Rehearse", "Quiz"],
                key="nd_selection_mode",
                on_change=handle_nd_settings_change,
            )
            st.selectbox(
                "Question Focus",
                [
                    "All",
                    "Z-Scores",
                    "Probabilities (%)",
                    "Inverse Normal (Find X)",
                    "Mean & Std Dev",
                    "Expected Value",
                ],
                key="nd_focus",
                on_change=handle_nd_settings_change,
            )
            st.toggle(
                "Overlay Normal Curve",
                key="nd_show_curve",
                on_change=lambda: st.session_state.update(
                    nd_problem_image=draw_normal_problem_image(
                        st.session_state.nd_problem_data,
                        show_curve=st.session_state.nd_show_curve,
                    )
                )
                if "nd_problem_data" in st.session_state
                else None,
            )
            st.radio(
                "Camera Mode",
                ["None", "App", "Native"],
                key="nd_camera_mode",
                horizontal=True,
                on_change=handle_nd_settings_change,
            )
            st.toggle(
                "Canvas Controls",
                key="nd_show_controls",
                on_change=handle_nd_settings_change,
            )

    # Quiz Mode Start / Completion Gate
    if st.session_state.nd_selection_mode == "Quiz" and not st.session_state.nd_quiz_active:
        if st.session_state.nd_quiz_completed:
            st.success(f"🎉 Quiz Complete! You completed {st.session_state.nd_questions_answered} questions.")
            if st.button("Restart Quiz", use_container_width=True, key="nd_restart_quiz"):
                handle_nd_settings_change()
                st.rerun()
            return

        st.info("Ready to test your Normal Distribution skills? This quiz covers all selected question types.")
        if st.button("🚀 Start Quiz", use_container_width=True, type="primary", key="nd_start_quiz"):
            st.session_state.nd_quiz_active = True
            st.session_state.nd_questions_answered = 0
            st.session_state.nd_generating = True
            st.rerun()
        return

    if st.session_state.nd_selection_mode == "Quiz" and st.session_state.nd_quiz_active:
        st.caption(f"Question **{st.session_state.nd_questions_answered + 1}**")
        st.divider()

    # Generate problem & image
    if st.session_state.nd_generating or "nd_problem_data" not in st.session_state:
        st.session_state.nd_generating = True
        with st.spinner("Generating distribution..."):
            p_data = get_next_nd_question()
            if p_data is None:
                st.session_state.nd_quiz_active = False
                st.rerun()

            st.session_state.nd_problem_data = p_data
            st.session_state.nd_problem_image = draw_normal_problem_image(
                p_data,
                show_curve=st.session_state.get("nd_show_curve", False),
                width_px=420,
                height_px=480,
            )
            st.session_state.nd_generating = False
            st.rerun()
    else:
        p_data = st.session_state.nd_problem_data
        bg_image = st.session_state.nd_problem_image

        problem_context = (
            f"NCEA Level 2 Normal Distribution Question:\n{p_data['text']}\n"
            f"Grading Rule: {p_data['grading_rule']}"
        )

        ai_marking_component.render_grading_suite(
            bg_image=bg_image,
            height_px=480,
            key_prefix=f"nd_suite_{st.session_state.nd_suite_refresh_id}",
            solution_requirement="general",
            problem_context=problem_context,
            show_controls=st.session_state.get("nd_show_controls", False),
            camera_mode=st.session_state.get("nd_camera_mode", "App"),
        )

        st.markdown('<div id="next-problem-btn"></div>', unsafe_allow_html=True)
        if st.button("Next Question", use_container_width=True, key="nd_next_btn"):
            if st.session_state.nd_selection_mode == "Cyclical":
                st.session_state.nd_question_queue.append(p_data["qtype"])
            if st.session_state.nd_selection_mode == "Quiz":
                st.session_state.nd_questions_answered += 1
            st.session_state.nd_generating = True
            st.session_state.nd_suite_refresh_id += 1
            st.rerun()