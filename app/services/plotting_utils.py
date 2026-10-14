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

# planner results plotting
def _coverage_from_results(res):
    """
    Rebuild, from the planner results only, what the plot needs:
        D           (n_rows, n_bins) demand grid padded with zeros over the overhang bins
        cov         (n_rows, n_bins) bool, bins covered by a shift (from server_schedule)
        blocks      [(row, start, end), ...] one per shift (datetimes)
        time_values pd.Series of bin starts (day + overhang), from df_curves
    """
    time_values = pd.to_datetime(res.df_curves["time"]).reset_index(drop=True)
    D = np.asarray(res.demand_grid, dtype=np.float64)
    extra = len(time_values) - D.shape[1]                       # bins after the end of the day
    if extra < 0:
        raise ValueError("df_curves is shorter than demand_grid")
    D = np.pad(D, ((0, 0), (0, extra)))
 
    row_of = {label: c for c, label in enumerate(res.server_labels)}
    cov = np.zeros(D.shape, dtype=bool)
    blocks = []
    for label, intervals in res.server_schedule.items():
        if label not in row_of:
            raise ValueError(f"schedule has server {label!r} that is not in server_labels")
        c = row_of[label]
        for start, end in intervals:
            in_shift = (time_values >= pd.Timestamp(start)) & (time_values < pd.Timestamp(end))
            cov[c] |= in_shift.to_numpy()
            blocks.append((c, start, end))

    total = float(D.sum())
    C = float(D[cov].sum())
    I = float((1.0 - D)[cov].sum()) 
    obj = C - res.alpha * I
    coverage = C / total if total > 0 else 1.0

    return D, cov, obj, coverage, blocks, time_values
 
def plot_allocation(res, show_second=True):
 
    D, cov, obj, coverage, blocks, time_values = _coverage_from_results(res)
    n_rows, n_bins = D.shape
 
    x0 = mdates.date2num(time_values)                          # left edge of each bin
    width = res.bin_minutes / (24 * 60)                        # bin width in matplotlib date units
    x_edges = np.append(x0, x0[-1] + width)                    # bin edges, for the step curves
    GREY, ORANGE, RED, WHITE = "grey", "orange", "red", "white"
    edge = dict(edgecolor="lightgray", linewidth=0.3)
 
    # top: demand vs allocation per server; bottom: arrivals vs processing per bin (same x axis)

    if show_second:
        fig, (ax, ax_curves) = plt.subplots(
            2, 1, figsize=(10, 0.3 * n_rows + 3), sharex=True,
            gridspec_kw={"height_ratios": [1.5, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 0.3 * n_rows + 1))
        ax_curves = None

    for c in range(n_rows):
        d, covered = D[c], cov[c]
        m = d > 1e-9
        ax.bar(x0[m], d[m], width=width, bottom=c, align="edge",
               color=np.where(covered[m], GREY, ORANGE), **edge)
        idle = 1.0 - d
        m = idle > 1e-9
        ax.bar(x0[m], idle[m], width=width, bottom=c + d[m], align="edge",
               color=np.where(covered[m], RED, WHITE), **edge)
 
    for c, start, end in blocks:
        xs, xe = mdates.date2num(start), mdates.date2num(end)
        ax.add_patch(Rectangle((xs, c), (xe - xs), 1, fill=False,
                               edgecolor="black", linewidth=1.6, zorder=3))
 
    ax.set_ylim(0, n_rows)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([
        f"server {res.server_labels[c]}" if c < len(res.server_labels) else "no server"
        for c in range(n_rows)
    ])
    ax.legend(
        handles=[
            Patch(facecolor=ORANGE, edgecolor="lightgray", label="uncovered"),
            Patch(facecolor=RED, edgecolor="lightgray", label="overshoot"),
            Patch(facecolor=GREY, edgecolor="lightgray", label="covered"),
        ],
        loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
    )

    if show_second:
        # bottom panel: step curves straight from df_curves (pax per bin)
        ax_curves.stairs(res.df_curves["demand"].to_numpy(), x_edges, color="tab:blue", linewidth=1.5, label="arrivals")
        ax_curves.stairs(res.df_curves["processing"].to_numpy(), x_edges, color="tab:green", linewidth=1.5, label="processing")
        ax_curves.set_ylim(bottom=0)
        ax_curves.set_ylabel(f"pax/{res.bin_minutes:g} mins")
        ax_curves.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    
        ax_curves.set_xlim(x0[0], x0[-1] + width)
        ax_curves.xaxis_date()
        ax_curves.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax_curves.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_curves.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))

    else:
        ax.set_xlim(x0[0], x0[-1] + width)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))

    title = f"L = {res.L:g} h  |  alpha = {res.alpha:g} | {len(blocks)} blocks  | obj={obj:.2f} | coverage = {coverage:.1%}"
    fig.suptitle(title)
    fig.autofmt_xdate(rotation=90)

    fig.canvas.draw()
    targets = [ax] + ([ax_curves] if show_second else [])
    for a in targets:
        a.set_axisbelow(True)
        a.grid(axis="x", which="major", linestyle="-", alpha=0.5, zorder=1)

    fig.tight_layout()
    plt.show()

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