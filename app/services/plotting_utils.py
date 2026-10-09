from matplotlib import pyplot as plt
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.dates as mdates
from app.services import utils
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

def plot_kde_fit(
    empirical_arrivals: pd.Series,
    kde_specs: dict,
    bw_methods=("scott", 0.2, 0.3, 0.5),
    bins=60,
    n_points=1000,
    ):

    arr = pd.to_numeric(
        pd.Series(empirical_arrivals),
        errors="coerce"
    ).dropna()

    shift = float(kde_specs.get("shift", 0))
    x_min = float(kde_specs["min_minutes"])
    x_max = float(kde_specs["max_minutes"])

    arr = arr + shift
    arr = arr[arr.between(x_min, x_max)]

    if len(arr) < 2 or arr.nunique() < 2:
        raise ValueError("Not enough variable observations to fit a KDE.")

    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(8, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]}
    )

    # ---------- PDF ----------
    ax1.hist(
        arr,
        bins=bins,
        density=True,
        alpha=0.35,
        label="PDF"
    )

    # ---------- CDF ----------
    arr_sorted = np.sort(arr)
    ecdf = np.arange(1, len(arr_sorted) + 1) / len(arr_sorted)

    ax2.step(
        arr_sorted,
        ecdf,
        where="post",
        color="k",
        linewidth=2,
        label="CDF"
    )

    for bw in bw_methods:
        kde = gaussian_kde(arr.to_numpy(), bw_method=bw)

        pdf = kde(x)
        pdf /= np.trapz(pdf, x)          # normalize

        cdf = np.cumsum(pdf) * dx
        cdf /= cdf[-1]                   # ensure last value = 1

        ax1.plot(
            x,
            pdf,
            linewidth=2,
            label=f"KDE bw={bw}"
        )

        ax2.plot(
            x,
            cdf,
            linewidth=2,
            label=f"KDE bw={bw}"
        )

    ax1.set_ylabel("Density")
    ax1.set_title("Empirical distribution and fitted KDE")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Advance [minutes]")
    ax2.set_ylabel("CDF")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_lanes_allocation(results, L, scale, alpha):
    df_alloc = results.df_alloc
    df_requirements = results.df_requirements
    df_arrivals = results.df_arrivals.copy()

    if df_requirements.empty:
        raise ValueError("df_requirements is empty.")

    fig, ax = plt.subplots(figsize=(14, 5))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    requirement_times = (
        pd.to_datetime(df_requirements["time"])
        .reset_index(drop=True)
    )

    n_timebins = len(df_requirements)

    if len(requirement_times) > 1:
        bin_duration_seconds = (
            requirement_times.iloc[1]
            - requirement_times.iloc[0]
        ).total_seconds()
    else:
        bin_duration_seconds = 15 * 60

    base_time = requirement_times.iloc[0]

    def time_to_x(values):
        times = pd.DatetimeIndex(pd.to_datetime(values))

        return (
            (times - base_time).total_seconds()
            / bin_duration_seconds
        )

    # Profilo degli arrivi
    arrivals_ax = ax.twinx()
    arrivals_ax.set_facecolor("white")

    df_arrivals["time"] = pd.to_datetime(df_arrivals["time"])

    all_arrivals = (
        df_arrivals.loc[
            df_arrivals["aggregation_level"] == "all"
        ]
        .sort_values("time")
    )

    if all_arrivals.empty:
        raise ValueError(
            "Aggregated arrival profile 'all' not found."
        )

    arrivals_x = time_to_x(all_arrivals["time"]) - 0.5

    arrivals_ax.step(
        arrivals_x,
        all_arrivals["median"].to_numpy(dtype=float),
        where="post",
        color="tab:green",
        linewidth=2,
        label="All arrivals: median",
    )

    arrivals_ax.set_ylabel(
        "Passenger arrivals per interval"
    )
    arrivals_ax.set_ylim(bottom=0)
    arrivals_ax.grid(False)

    arrivals_ax.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_alpha(0)

    # Rettangoli dei turni allocati
    for _, row in df_alloc.iterrows():
        ax.add_patch(
            Rectangle(
                (
                    row["x_start"] - 0.5,
                    row["y"] - 0.5,
                ),
                row["x_end"] - row["x_start"],
                1,
                facecolor="lightgray",
                edgecolor="black",
                linewidth=1.5,
                alpha=0.65,
            )
        )

    # Profilo del numero di varchi allocati
    optimized_nlanes = (
        df_requirements["optimized_nlanes"]
        .to_numpy(dtype=int)
    )

    x_edges = np.arange(n_timebins + 1) - 0.5

    allocated_plot = np.r_[
        optimized_nlanes,
        optimized_nlanes[-1],
    ]

    ax.step(
        x_edges,
        allocated_plot - 0.5,
        where="post",
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label="Allocated lanes",
    )

    ax.set_xlim(-0.5, n_timebins - 0.5)

    # Numero di righe effettivamente utilizzate in df_alloc
    n_allocation_rows = (
        int(df_alloc["y"].max()) + 1
        if not df_alloc.empty
        else 0
    )

    max_lanes = max(
        1,
        n_allocation_rows,
        int(df_requirements["optimized_nlanes"].max()),
    )

    ax.set_ylim(-0.5, max_lanes - 0.5)

    # Griglia verticale: un bordo per ogni intervallo
    cell_edges = np.arange(n_timebins + 1) - 0.5

    ax.set_xticks(
        cell_edges,
        minor=True,
    )

    # Etichetta temporale ogni ora, assumendo intervalli da 15 minuti
    bins_per_hour = max(
        1,
        round(3600 / bin_duration_seconds),
    )

    major_tick_idx = np.arange(
        0,
        n_timebins,
        bins_per_hour,
    )

    major_tick_labels = (
        requirement_times.iloc[major_tick_idx]
        .dt.strftime("%H:%M")
    )

    ax.set_xticks(major_tick_idx)
    ax.set_xticklabels(
        major_tick_labels,
        rotation=45,
        ha="right",
    )

    ax.grid(
        which="minor",
        axis="x",
        color="lightgray",
        linestyle="-",
        linewidth=0.8,
        alpha=0.8,
    )

    # Griglia orizzontale
    lane_edges = np.arange(max_lanes + 1) - 0.5

    ax.set_yticks(
        lane_edges,
        minor=True,
    )

    ax.grid(
        which="minor",
        axis="y",
        color="lightgray",
        linestyle="-",
        linewidth=0.8,
        alpha=0.8,
    )

    ax.set_axisbelow(True)

    title = "Allocated lanes and passenger arrivals"

    if (scale is not None and alpha is not None
    ):
        title += (
            f" (L: {L:.1f}, "
            f"scale: {scale:.1f}, "
            f"alpha: {alpha:.1f})"
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Lane index / lane count")

    # Legenda combinata
    lanes_handles, lanes_labels = (
        ax.get_legend_handles_labels()
    )

    arrivals_handles, arrivals_labels = (
        arrivals_ax.get_legend_handles_labels()
    )

    ax.legend(
        lanes_handles + arrivals_handles,
        lanes_labels + arrivals_labels,
        loc="upper right",
    )

    plt.tight_layout()
    plt.show()

def plot_opt_results(df_requirements, df_arrivals, server_schedule, server_registry):
    df_arr = (
        df_arrivals[df_arrivals["aggregation_level"] == "all"]
        .sort_values("time")
    )
    df_req = df_requirements.sort_values("time")

    server_labels = utils.sort_labels_by_priority(server_registry)
    n_labels = len(server_labels)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 4 + 0.3 * n_labels), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # --- top: arrivals vs processing rate ---
    ax1.fill_between(
        df_arr["time"], df_arr["p5"], df_arr["p95"],
        color="tab:blue", alpha=0.12
    )

    ax1.plot(
        df_arr["time"], df_arr["median"],
        color="tab:blue", linewidth=1.8, label="arrivals (median)",
    )

    ax1.step(
        df_req["time"], df_req["processing_rate"],
        where="post", color="tab:red", linewidth=1.8,
        linestyle="--", label="processing_rate (pax/15min)",
    )

    ax1.set_ylabel("pax / 15min")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    # --- bottom: gantt per server_label ---
    for row, label in enumerate(server_labels):
        periods = server_schedule.get(label, [])
        bars = [
            (mdates.date2num(start), mdates.date2num(end) - mdates.date2num(start))
            for start, end in periods
        ]
        ax2.broken_barh(bars, (row - 0.4, 0.8), facecolors="grey", edgecolor="darkgrey", linewidth=2)

    ax2.set_yticks(range(n_labels))
    ax2.set_yticklabels(server_labels)
    ax2.set_ylim(-0.5, n_labels - 0.5)
    ax2.invert_yaxis()
    ax2.set_ylabel("server")
    ax2.grid(axis="x", alpha=0.25)

    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax2.tick_params(axis="x", which="minor", length=3)
    ax2.set_xlabel("time")

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def plot_grid_overlay(grid, rectangles, time_values, bin_minutes=None):
    grid = np.asarray(grid)
    alloc_grid = build_alloc_grid(rectangles, grid.shape)
    time_values = pd.to_datetime(pd.Series(time_values).reset_index(drop=True))

    n_rows, n_bins = grid.shape

    if bin_minutes is None:
        bin_minutes = (
            (time_values.iloc[1] - time_values.iloc[0]).total_seconds() / 60
            if len(time_values) > 1
            else 15
        )

    category = grid.astype(np.uint8) + 2 * alloc_grid.astype(np.uint8)

    colors = ["white", "orange", "red", "grey"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    edges = list(time_values) + [time_values.iloc[-1] + pd.Timedelta(minutes=bin_minutes)]
    x_edges = mdates.date2num(edges)
    y_edges = np.arange(n_rows + 1)

    fig, ax = plt.subplots(figsize=(14, 0.4 * n_rows + 2))

    ax.pcolormesh(x_edges, y_edges, category, cmap=cmap, norm=norm, edgecolors="lightgray", linewidth=0.3)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([f"riga {y}" for y in range(n_rows)])

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    ax.set_xlabel("time")
    ax.set_title("Domanda vs allocazione")

    legend_handles = [
        Patch(facecolor="orange", edgecolor="lightgray", label="domanda scoperta"),
        Patch(facecolor="red", edgecolor="lightgray", label="overshoot (spreco)"),
        Patch(facecolor="grey", edgecolor="lightgray", label="copertura corretta"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def _step(ax, x, y, **kw):
    return ax.step(x, y, where="post", **kw)

def plot_all(df, df_kpis=None, date=None):

    # filter dataframes by date
    if date is not None:
        date_norm = pd.to_datetime(date, dayfirst=True).normalize()
        df_timeline = df[df['date'].dt.normalize() == date_norm].copy()
        if df_kpis is None:
            df_kpis = pd.DataFrame([{
                'date': pd.to_datetime(date).normalize() if date is not None else pd.NaT,
                'qtime_p50': np.nan,
                'queue_length_p50': np.nan,
                'penalized_system_pct': np.nan,
                'ttd_below_threshold_pct': np.nan,
                'avg_utilization_pct': np.nan,
                'full_utilization_pct': np.nan
            }])
        df_kpis = df_kpis[df_kpis['date'].dt.normalize() == date_norm].copy()
    else:
        # raise ValueError("Date is required for plotting")
        df_timeline = df.copy()
        if df_kpis is None:
            df_kpis = pd.DataFrame([{
                'date': pd.to_datetime(date).normalize() if date is not None else pd.NaT,
                'qtime_p50': np.nan,
                'queue_length_p50': np.nan,
                'penalized_system_pct': np.nan,
                'ttd_below_threshold_pct': np.nan,
                'avg_utilization_pct': np.nan,
                'full_utilization_pct': np.nan
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
        arrival_rate_q25 = df_timeline['arrival_rate_p5']
        arrival_rate_q75 = df_timeline['arrival_rate_p95']
    else:
        arrival_rate_q25 = None
        arrival_rate_q75 = None

    theo_processing_rate = df_timeline['theoretical_processing_rate']

    # processing rates
    if 'processing_rate_median' in df_timeline.columns:
        processing_rate_median = df_timeline['processing_rate_median']
    else:
        processing_rate_median = df_timeline['processing_rate']

    if 'processing_rate_p5' in df_timeline.columns and 'processing_rate_p95' in df_timeline.columns:
        processing_rate_q25 = df_timeline['processing_rate_p5']
        processing_rate_q75 = df_timeline['processing_rate_p95']
    else:
        processing_rate_q25 = None
        processing_rate_q75 = None

    # queue lengths
    if 'queue_length_median' in df_timeline.columns:
        queue_median = df_timeline['queue_length_median']
    else:
        queue_median = df_timeline['queue_length']

    if 'queue_length_p5' in df_timeline.columns and 'queue_length_p95' in df_timeline.columns:
        queue_q25 = df_timeline['queue_length_p5']
        queue_q75 = df_timeline['queue_length_p95']
    else:
        queue_q25 = None
        queue_q75 = None

    # server occupation
    if 'occupied_servers_median' in df_timeline.columns:
        server_median = df_timeline['occupied_servers_median']
    else:
        server_median = df_timeline['occupied_servers']

    if 'occupied_servers_p5' in df_timeline.columns and 'occupied_servers_p95' in df_timeline.columns:
        server_q25 = df_timeline['occupied_servers_p5']
        server_q75 = df_timeline['occupied_servers_p95']
    else:
        server_q25 = None
        server_q75 = None

    # capacity
    capacity = df_timeline['capacity']

    # expected queue time
    if 'expected_qtime_min_median' in df_timeline.columns:
        expected_qtime_median = df_timeline['expected_qtime_min_median']
        expected_qtime_p25 = df_timeline['expected_qtime_min_p25']
        expected_qtime_p75 = df_timeline['expected_qtime_min_p75']
    else:
        expected_qtime_median = df_timeline['expected_qtime_min']
        expected_qtime_p25 = None
        expected_qtime_p75 = None

    # Plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios':[1,1,1,1]})

    add_kpi_cells(fig, df_kpis)

    # plot rates
    line1, = _step(ax1, x, arrival_rate_median.values, color='blue', alpha=0.5, linestyle='-', label='Arrival Rate')

    if arrival_rate_q25 is not None and arrival_rate_q75 is not None:
        ax1.fill_between(x, arrival_rate_q25, arrival_rate_q75, step="post",
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

    if queue_q25 is not None and queue_q75 is not None:
        ax2.fill_between(x, queue_q25, queue_q75, step="post",
                         color='royalblue', alpha=0.2)
    ax2.set_ylabel('# of pax')
    ax2b = ax2.twinx()
    line2,= _step(ax2b, x, processing_rate_median.values, color='orange', label='Processing Rate')
    
    if processing_rate_q25 is not None and processing_rate_q75 is not None:
        ax2b.fill_between(x, processing_rate_q25, processing_rate_q75, step="post",
                        color='orange', alpha=0.2)
    ax2b.set_ylabel('Processing rate')
    
    ax2.set_xlabel('Time')
    ax2.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
    ax2.set_ylim(bottom=0)
    ax2b.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc='upper left')

    # plot servers
    line1,= _step(ax3, x, server_median.values, color='salmon', alpha=0.7, linestyle='-', label='Busy Servers')
    if server_q25 is not None and server_q75 is not None:
        ax3.fill_between(x, server_q25, server_q75, step="post",
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
    if expected_qtime_p25 is not None and expected_qtime_p75 is not None:
        ax4.fill_between(x, expected_qtime_p25, expected_qtime_p75, step="post",
                         color='salmon', alpha=0.2)

    ax4.set_ylabel('Minutes')
    ax4.grid(axis='x', linestyle='-', alpha=0.5, zorder=1)
    ax4.tick_params(axis='y')
    ax4.set_ylim(bottom=0)
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
    lines = [line1]
    labels = [line.get_label() for line in lines]
    ax4.legend(lines, labels, loc='upper left')

    min_time = x.min()
    max_time = x.max()

    if max_time - min_time <= pd.Timedelta('4h'):
        freq = '30min'
    elif max_time - min_time <= pd.Timedelta('12h'):
        freq = '60min'
    else:
        freq = '180min'
    xticks = pd.date_range(start=min_time.ceil(freq), end=max_time, freq=freq)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(axis='x', labelbottom=True)
    fig.subplots_adjust(top=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def add_kpi_cells(fig, df_kpis):

    labels = ['qtime_p50', 'queue_length_p50', 'penalized_system_pct', 'ttd_below_threshold_pct', 'avg_utilization_pct', 'full_utilization_pct']
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