from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.dates as mdates
from app.services import utils
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle, Patch
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import json

# planner results plotting
def coverage_mask(blocks, labels, shape):
    """Boolean (n_rows, T) mask of covered cells."""
    row_of = {l: c for c, l in enumerate(labels)}
    cov = np.zeros(shape, dtype=bool)
    for label, bl in blocks.items():
        if label not in row_of:
            raise ValueError(f"label {label!r} not in labels")
        for s, l in bl:
            if s < 0 or s + l > shape[1]:
                raise ValueError(f"block ({s},{l}) of {label!r} exceeds the grid (T={shape[1]})")
            cov[row_of[label], s:s + l] = True
    return cov

def coverage_from_arrays(D, blocks, labels, alpha):
    """-> cov mask, objective = covered - alpha*idle, coverage = covered/total."""
    D = np.asarray(D, dtype=np.float64)
    cov = coverage_mask(blocks, labels, D.shape)
    total = float(D.sum())
    C = float(D[cov].sum())
    I = float((1.0 - D)[cov].sum())
    return cov, C - alpha * I, (C / total if total > 0 else 1.0)
 
def processing_from_blocks(blocks, rate_by_label, T):
    """Capacity per bin (pax/bin) = sum of the rates of the open lanes."""
    out = np.zeros(T)
    for label, bl in blocks.items():
        for s, l in bl:
            out[s:s + l] += rate_by_label[label]
    return out

def schedule_to_slot_blocks(server_schedule, time_values):
    """{label: [(start_dt, end_dt)]} -> {label: [(start_slot, length_slots)]} on time_values."""
    tv = pd.DatetimeIndex(pd.to_datetime(time_values))
    out = {}
    for label, ivs in server_schedule.items():
        bl = []
        for s, e in ivs:
            i = int(tv.searchsorted(pd.Timestamp(s), side="left"))
            j = int(tv.searchsorted(pd.Timestamp(e), side="left"))
            bl.append((i, j - i))
        out[label] = sorted(bl)
    return out

def parse_blocks(x):
    """JSON string or dict -> {label: [(start, length)]}"""
    d = json.loads(x) if isinstance(x, str) else x
    return {k: [tuple(int(v) for v in b) for b in bl] for k, bl in d.items()}
 
def shifts_label(blocks, bin_minutes):
    lens = pd.Series([l * bin_minutes / 60 for bl in blocks.values() for _, l in bl]).round(2)
    counts = lens.value_counts().sort_index()
    return " + ".join(f"{n}×{h:g}h" for h, n in counts.items()) or "no shifts"
 
def plot_blocks(D, blocks, labels, time_values, alpha, bin_minutes,
                metrics=None, demand_curve=None, processing_curve=None,
                show_second=True, title_prefix=None):
    D = np.asarray(D, dtype=np.float64)
    labels = list(labels)
    time_values = pd.to_datetime(pd.Series(list(time_values))).reset_index(drop=True)
    n_rows, T = D.shape
    if len(time_values) != T:
        raise ValueError(f"time_values has {len(time_values)} entries, D has {T} bins")
 
    cov, obj, coverage = coverage_from_arrays(D, blocks, labels, alpha)
    if metrics is not None:                                   # stored values win (title consistent with the df)
        obj = metrics.get("objective", obj)
        coverage = metrics.get("coverage", coverage)
 
    flat = [(labels.index(l), s, ln) for l, bl in blocks.items() for s, ln in bl]
    multi_length = len({ln for _, _, ln in flat}) > 1
    show_second = show_second and demand_curve is not None and processing_curve is not None
 
    x0 = mdates.date2num(time_values)
    width = bin_minutes / (24 * 60)
    x_edges = np.append(x0, x0[-1] + width)
    edge = dict(edgecolor="lightgray", linewidth=0.3)
 
    if show_second:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 0.35 * n_rows + 5), sharex=True,
                                      gridspec_kw={"height_ratios": [max(2, 0.35 * n_rows), 2]})
    else:
        fig, ax = plt.subplots(figsize=(10, 0.35 * n_rows + 2)); ax2 = None
 
    for c in range(n_rows):
        d, k = D[c], cov[c]
        ax.bar(x0, np.where(k, d, 0), width=width, bottom=c, align="edge", color='GREY', **edge)
        ax.bar(x0, np.where(k, 1 - d, 0), width=width, bottom=c + np.where(k, d, 0), align="edge", color='RED', **edge)
        ax.bar(x0, np.where(~k, d, 0), width=width, bottom=c, align="edge", color='ORANGE', **edge)
        ax.bar(x0, np.where(~k, 1 - d, 0), width=width, bottom=c + np.where(~k, d, 0), align="edge", color='WHITE', **edge)
    for c, s, ln in flat:
        ax.add_patch(Rectangle((x_edges[s], c), x_edges[s + ln] - x_edges[s], 1,
                               fill=False, edgecolor="black", linewidth=1.6))
        if multi_length:
            ax.text((x_edges[s] + x_edges[s + ln]) / 2, c + 0.5, f"{ln * bin_minutes / 60:g}h",
                    ha="center", va="center", fontsize=8)
    ax.set_ylim(0, n_rows); ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([f"server {l}" for l in labels])
    ax.set_xlabel("Time"); ax.set_ylabel("Server")
    ax.legend(handles=[Patch(color='GREY', label="demand covered"), Patch(color='RED', label="idle capacity"),
                       Patch(color='ORANGE', label="demand uncovered")], loc="upper right", fontsize=8)
 
    if ax2 is not None:
        ax2.stairs(np.asarray(demand_curve) / bin_minutes, x_edges, label="demand", color="tab:blue")
        ax2.stairs(np.asarray(processing_curve) / bin_minutes, x_edges, label="processing", color="tab:green")
        ax2.set_ylabel("pax / min"); ax2.legend(fontsize=8)
 
    axes = [ax] + ([ax2] if ax2 is not None else [])
    for a in axes:
        a.xaxis.set_major_locator(mdates.HourLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
        a.grid(axis="x", which="major", alpha=0.4)
    title = (f"L: {shifts_label(blocks, bin_minutes)} |  alpha = {alpha:g}  | "
             f"obj={obj:.2f} | coverage = {coverage:.2%}")
    fig.suptitle(f"{title_prefix} — {title}" if title_prefix else title)
    fig.autofmt_xdate(rotation=90)
    fig.tight_layout()
    plt.show()
 
def plot_allocation(res, show_second=True):
    """Backward-compatible wrapper: only extracts the minimal inputs from `res`."""
    tv = pd.to_datetime(res.df_curves["time"]).reset_index(drop=True)
    D = np.asarray(res.demand_grid, dtype=np.float64)
    tv = tv.iloc[:D.shape[1]]
    blocks = schedule_to_slot_blocks(res.server_schedule, tv)
    plot_blocks(D, blocks, res.server_labels, tv, res.alpha, res.bin_minutes,
                demand_curve=res.df_curves["demand"].to_numpy()[:D.shape[1]],
                processing_curve=res.df_curves["processing"].to_numpy()[:D.shape[1]],
                show_second=show_second)
 
def plot_run_from_row(row, D, labels, time_values, bin_minutes, demand_curve=None,
                      rate_by_label=None, show_second=True):
    """
    Plot a df_pert row: blocks come from row['blocks'], alpha/objective/coverage from the row itself.
    D, labels, time_values, demand_curve: from the (single) demand the experiment was run on.
    rate_by_label (planner.rate_by_label): needed only to draw the processing curve.
    """
    blocks = parse_blocks(row["blocks"])
    D = np.asarray(D, dtype=np.float64)

    proc = (processing_from_blocks(blocks, rate_by_label, D.shape[1])
            if rate_by_label is not None else None)
    plot_blocks(D, blocks, labels, time_values, row["alpha"], bin_minutes,
                    metrics={"objective": row["objective"], "coverage": row["coverage"]},
                    demand_curve=demand_curve, processing_curve=proc, show_second=show_second,
                    title_prefix=f"rep {row['rep']}")

# simulation results plotting
def _step(ax, x, y, **kw):
    return ax.step(x, y, where="post", **kw)

def plot_simulation(df, df_kpis=None, date=None):

    # filter dataframes by date
    if date is not None:
        date_norm = pd.to_datetime(date, dayfirst=True).normalize()
        df_timeline = df[df['date'].dt.normalize() == date_norm].copy()
        if df_kpis is None:
            df_kpis = pd.DataFrame([{
                'date': pd.to_datetime(date).normalize() if date is not None else pd.NaT,
                'qtime_p50': np.nan,
                'n_unserved': np.nan,
                'qtime_above_15_pct': np.nan,
                'qtime_above_30_pct': np.nan,
                'qtime_above_60_pct': np.nan,
            }])
        df_kpis = df_kpis[df_kpis['date'].dt.normalize() == date_norm].copy()
    else:
        # raise ValueError("Date is required for plotting")
        df_timeline = df.copy()
        if df_kpis is None:
            df_kpis = pd.DataFrame([{
                'date': pd.to_datetime(date).normalize() if date is not None else pd.NaT,
                'qtime_p50': np.nan,
                'n_unserved': np.nan,
                'qtime_above_15_pct': np.nan,
                'qtime_above_30_pct': np.nan,
                'qtime_above_60_pct': np.nan,
            }])
        df_kpis = df_kpis.copy()
    
    if df_timeline.empty:
        print(f"No data available for the date: {date}")
        return plt.figure()
    
    x = df_timeline['time_bin']

    # arrival rates
    if 'arrival_rate_median' in df_timeline.columns:
        arrival_rate_median = df_timeline['arrival_rate_median']
    else:
        arrival_rate_median = df_timeline['arrival_count']

    if 'arrival_rate_p5' in df_timeline.columns and 'arrival_rate_p95' in df_timeline.columns:
        arrival_rate_q5 = df_timeline['arrival_rate_p5']
        arrival_rate_q95 = df_timeline['arrival_rate_p95']
    else:
        arrival_rate_q5 = None
        arrival_rate_q95 = None

    theo_processing_rate = df_timeline['theoretical_processing_rate']

    # processing rates
    if 'processing_rate_median' in df_timeline.columns:
        processing_rate_median = df_timeline['processing_rate_median']
    else:
        processing_rate_median = df_timeline['processing_rate']

    if 'processing_rate_p5' in df_timeline.columns and 'processing_rate_p95' in df_timeline.columns:
        processing_rate_q5 = df_timeline['processing_rate_p5']
        processing_rate_q95 = df_timeline['processing_rate_p95']
    else:
        processing_rate_q5 = None
        processing_rate_q95 = None

    # queue lengths
    if 'queue_length_median' in df_timeline.columns:
        queue_median = df_timeline['queue_length_median']
    else:
        queue_median = df_timeline['queue_length']

    if 'queue_length_p5' in df_timeline.columns and 'queue_length_p95' in df_timeline.columns:
        queue_q5 = df_timeline['queue_length_p5']
        queue_q95 = df_timeline['queue_length_p95']
    else:
        queue_q5 = None
        queue_q95 = None

    # server occupation
    if 'occupied_servers_median' in df_timeline.columns:
        server_median = df_timeline['occupied_servers_median']
    else:
        server_median = df_timeline['occupied_servers']

    if 'occupied_servers_p5' in df_timeline.columns and 'occupied_servers_p95' in df_timeline.columns:
        server_q5 = df_timeline['occupied_servers_p5']
        server_q95 = df_timeline['occupied_servers_p95']
    else:
        server_q5 = None
        server_q95 = None

    # capacity
    capacity = df_timeline['capacity']

    # expected queue time
    if 'expected_qtime_min_median' in df_timeline.columns:
        expected_qtime_median = df_timeline['expected_qtime_min_median']
        expected_qtime_p5 = df_timeline['expected_qtime_min_p5']
        expected_qtime_p95 = df_timeline['expected_qtime_min_p95']
    else:
        expected_qtime_median = df_timeline['expected_qtime_min']
        expected_qtime_p5 = None
        expected_qtime_p95 = None

    # Plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios':[1,1,1,1]})

    add_kpi_cells(fig, df_kpis)

    # plot rates
    line1, = _step(ax1, x, arrival_rate_median.values, color='blue', alpha=0.5, linestyle='-', label='Arrival Rate')

    if arrival_rate_q5 is not None and arrival_rate_q95 is not None:
        ax1.fill_between(x, arrival_rate_q5, arrival_rate_q95, step="post",
                          color='blue', alpha=0.2)
        
    line2, = _step(ax1, x, theo_processing_rate.values, color='green', alpha=0.7, linestyle='-', label='Theoretical average processing Rate')
    ax1.set_ylabel('pax')
    ax1.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    # plot queue
    line1, = _step(ax2, x, queue_median.values, color='royalblue', label='Queue Length')

    if queue_q5 is not None and queue_q95 is not None:
        ax2.fill_between(x, queue_q5, queue_q95, step="post",
                         color='royalblue', alpha=0.2)
    ax2.set_ylabel('# of pax')
    ax2b = ax2.twinx()
    line2,= _step(ax2b, x, processing_rate_median.values, color='orange', label='Processing Rate')
    
    if processing_rate_q5 is not None and processing_rate_q95 is not None:
        ax2b.fill_between(x, processing_rate_q5, processing_rate_q95, step="post",
                        color='orange', alpha=0.2)
    ax2b.set_ylabel('Processing rate')
    
    ax2.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
    ax2.set_ylim(bottom=0)
    ax2b.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper left')

    # plot servers
    line1,= _step(ax3, x, server_median.values, color='salmon', alpha=0.7, linestyle='-', label='Busy Servers')
    if server_q5 is not None and server_q95 is not None:
        ax3.fill_between(x, server_q5, server_q95, step="post",
                         color='salmon', alpha=0.2)
    line2,= _step(ax3, x, capacity.values, color='red', alpha=0.7, linestyle='-', label='Capacity')
    ax3.set_ylabel('# servers')
    ax3.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
    ax3.tick_params(axis='y')
    ax3.set_ylim(bottom=0)
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc='upper left')

    # plot expected queue time
    line1,= _step(ax4, x, expected_qtime_median.values, color='salmon', alpha=0.7, linestyle='-', label='Expected Queue Time')
    if expected_qtime_p5 is not None and expected_qtime_p95 is not None:
        ax4.fill_between(x, expected_qtime_p5, expected_qtime_p95, step="post",
                         color='salmon', alpha=0.2)

    ax4.set_ylabel('Minutes')
    ax4.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
    ax4.tick_params(axis='y')
    ax4.set_ylim(bottom=0)
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
    lines = [line1]
    labels = [line.get_label() for line in lines]
    ax4.legend(lines, labels, loc='upper left')

    # time axis: same style as plot_allocation (sharex -> set it once, on the bottom panel)
    step = x.diff().dropna().median() if len(x) > 1 else pd.Timedelta(minutes=15)

    ax4.set_xlim(x.min(), x.max() + step)
    ax4.xaxis_date()
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax4.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))

    fig.autofmt_xdate(rotation=90)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def add_kpi_cells(fig, df_kpis):

    labels = ['qtime_p50', 'n_unserved', 'qtime_above_15_pct', 'qtime_above_30_pct', 'qtime_above_60_pct']
    values = [df_kpis[label].iloc[0] for label in labels]
    n = len(labels)

    left, right = 0.02, 0.98
    y = 0.98 
    gap = 0.008
    total_w = right - left
    w = (total_w - gap*(n-1)) / n

    for i, (lab, val) in enumerate(zip(labels, values)):
        x0 = left + i*(w + gap)
        if isinstance(val, (int, float)):
            if lab.endswith("pct"):
                txt_val = f"{val:.2f}%"
            else:
                txt_val = f"{val:.2f}"
        else:
            txt_val = f"{val}"

        txt = f"{lab}\n{txt_val}"
        fig.text(
            x0 + w/2, y,
            txt,
            ha="center", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7")
        )

def plot_queue_cdf(df_queue_cdf: pd.DataFrame) -> plt.Figure:
    """
    Plot the queue time CDF as inverse (x = % pax, y = time in minutes):
    per ogni frazione di passeggeri, il tempo massimo atteso.
    """
    x = np.concatenate([[0.0], df_queue_cdf["cum_pct"].to_numpy()])
    y = np.concatenate([[0.0], df_queue_cdf["queue_time_bin_end"].to_numpy()])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.step(x, y, where="post", color="tab:red", linewidth=1.8)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("% pax")
    ax.set_ylabel("queue time (min)")
    ax.set_title("Queue time CDF")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()
    
def plot_time_cdf(df_pax: pd.DataFrame, date=None) -> None:
    df = df_pax.copy()

    if df.empty:
        print("No passenger data available")
        return

    if date is not None:
        date_norm = pd.to_datetime(date, dayfirst=True).normalize()
        df = df[df['date'].dt.normalize() == date_norm].copy()

    if df.empty:
        print(f"No passenger data available for the date: {date}")
        return

    df = df[df['start_dt'].notna()].copy()

    plt.figure(figsize=(5, 4))

    medians = []

    for key, g in df.groupby('scenario_id'):
        waits = np.sort(g["queue_time"].to_numpy(dtype=float))

        if len(waits) == 0:
            continue

        p = np.arange(1, len(waits) + 1) / len(waits)

        plt.plot(waits, p, lw=1, alpha=0.7)
        medians.append(np.median(waits))

    if medians:
        mean_p50 = np.mean(medians)

        plt.axhline(
            0.5,
            color="red",
            linestyle="-",
            linewidth=0.5,
        )

        plt.scatter(
            [mean_p50],
            [0.5],
            s=25,
            zorder=10,
            color="red",
            label=f"Mean across scenarios: {mean_p50:.1f} min",
        )

    plt.xlabel("Waiting time [min]")
    plt.ylabel("Cumulative probability")
    plt.title("Waiting time CDF")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()