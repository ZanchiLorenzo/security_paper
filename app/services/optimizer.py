import pandas as pd
import math
import numpy as np
import bisect
import logging
logger = logging.getLogger(__name__)

def mask_to_values(mask: int, values: np.ndarray) -> np.ndarray:
    """
    Given a bitmask and an array of values, return the values corresponding to the set bits in the mask.
    """
    width = len(values)
    bits = np.array([(mask >> x) & 1 for x in range(width)], dtype=bool)
    return values[bits]

class LanesOptimizer:
    def __init__(self, arrival_scenarios_dict, mean_processing_rate):

        if not arrival_scenarios_dict:
            raise ValueError("arrival_scenarios_dict cannot be empty.")

        if mean_processing_rate <= 0:
            raise ValueError("mean_processing_rate must be > 0.")

        
        self.grid = self._base_grid
        self.arrival_curve = self._base_arrival_curve

    def optimize(self, L, scale, alpha):

        if L <= 0:
            raise ValueError("L must be > 0.")

        if scale <= 0:
            raise ValueError("scale must be > 0.")

        if alpha < 0:
            raise ValueError("alpha must be >= 0.")

        time_values = (
            self._base_arrival_curve["time"]
            .reset_index(drop=True)
        )

        bin_duration = (
            time_values.iloc[1] - time_values.iloc[0]
            if len(time_values) > 1
            else pd.Timedelta(minutes=15)
        )

        bin_hours = (bin_duration.total_seconds() / 3600)

        # conversion: hours to number of time cells
        L_cells = max(1, math.floor(L / bin_hours + 0.5))
        L_rounded_hours = L_cells * bin_hours

        pad_left = 0
        pad_right = max(0, L_cells - 1)

        if pad_right > 0:
            grid = np.pad(self._base_grid, ((0, 0), (pad_left, pad_right)), mode="constant")

            post_times = pd.Series(
                [time_values.iloc[-1] + (i + 1) * bin_duration for i in range(pad_right)]
            )
            time_values = pd.concat(
                [time_values, post_times], ignore_index=True
            )

            continuous_demand = np.concatenate([
                self._base_arrival_curve["required_counters_exact"].to_numpy(),
                np.zeros(pad_right),
            ])

            arrival_curve = pd.DataFrame({
                "time": time_values.to_numpy(),
                "arrivals": np.concatenate([
                    self._base_arrival_curve["arrivals"].to_numpy(),
                    np.zeros(pad_right, dtype=self._base_arrival_curve["arrivals"].dtype),
                ]),
                "required_counters_exact": continuous_demand,
                "required_counters": np.concatenate([
                    self._base_arrival_curve["required_counters"].to_numpy(),
                    np.zeros(pad_right, dtype=self._base_arrival_curve["required_counters"].dtype),
                ]),
            })
        else:
            grid = self._base_grid
            continuous_demand = self._base_arrival_curve["required_counters_exact"].to_numpy()
            arrival_curve = self._base_arrival_curve

        self.grid = grid
        self.arrival_curve = arrival_curve

        rectangles = self.allocate_shape(
            grid=grid,
            L=L_cells,
            total_rectangles = math.ceil(self.min_rectangles_needed(grid, L_cells) * scale),
            alpha=alpha,
            continuous_demand=continuous_demand
        )

        df_alloc = pd.DataFrame(rectangles)

        if not df_alloc.empty:
            time_arr = time_values.to_numpy()
            x_start = df_alloc["x_start"].to_numpy()
            x_end = df_alloc["x_end"].to_numpy()

            df_alloc["start_time"] = time_arr[x_start]

            in_range = x_end < len(time_values)
            end_time = np.empty(len(df_alloc), dtype=time_arr.dtype)
            end_time[in_range] = time_arr[x_end[in_range]]
            end_time[~in_range] = time_arr[-1] + bin_duration
            df_alloc["end_time"] = end_time

            df_alloc["duration_hours"] = L_rounded_hours

            values_in_range, mean_value, mean_value_inside, mean_value_outside = [], [], [], []

            for row in df_alloc.itertuples(index=False):
                row_grid = grid[row.y]
                range_values = continuous_demand[row.x_start:row.x_end]
                inside_bool = row_grid[row.x_start:row.x_end].astype(bool)

                values_in_range.append(range_values)
                mean_value.append(range_values.mean())
                mean_value_inside.append(range_values[inside_bool].mean() if inside_bool.any() else np.nan)
                mean_value_outside.append(range_values[~inside_bool].mean() if (~inside_bool).any() else np.nan)

            df_alloc["values"] = values_in_range
            df_alloc["mean_value"] = mean_value
            df_alloc["mean_value_inside"] = mean_value_inside
            df_alloc["mean_value_outside"] = mean_value_outside

        alloc_grid = np.zeros_like(grid)

        for rectangle in rectangles:
            y = rectangle["y"]
            x_start = rectangle["x_start"]
            x_end = rectangle["x_end"]

            alloc_grid[y, x_start:x_end] = 1

        df_requirements = pd.DataFrame({
            "time": time_values.to_numpy(),
            "timebin": np.arange(grid.shape[1]),
            "optimized_nlanes": alloc_grid.sum(axis=0).astype(int),
        })

        return df_alloc, df_requirements

    def get_runs(self, row):
        runs = []
        start = None

        for i, v in enumerate(row):
            if v == 1 and start is None:
                start = i
            elif v == 0 and start is not None:
                runs.append((start, i))
                start = None

        if start is not None:
            runs.append((start, len(row)))

        return runs

    def allocate_row_dp_time(self, value_row, Ls, max_rectangles, alpha):
        """
        DP che percorre l'asse temporale bin per bin: a ogni bin x decide se non
        aprire nulla, oppure aprire un turno con una delle lunghezze disponibili
        in Ls (es. [16, 8] per 4h/2h a bin da 15 min), poi prosegue.
        """

        width = len(value_row)
        K = max_rectangles
        NEG = float("-inf")

        prefix = [0] * (width + 1)
        for i, v in enumerate(value_row):
            prefix[i + 1] = prefix[i] + v

        # print('prefix', prefix)

        def inside_of(a, b):
            return prefix[b] - prefix[a]

        dp = [[NEG] * (K + 1) for _ in range(width + 1)]
        dp[0][0] = 0.0

        choice = [[None] * (K + 1) for _ in range(width + 1)]

        for x in range(width):
            for k in range(K + 1):
                cur = dp[x][k]
                if cur == NEG:
                    continue

                if cur > dp[x + 1][k]:
                    old = dp[x + 1][k]
                    dp[x + 1][k] = cur
                    choice[x + 1][k] = ("skip", x, k)
                    # print(f"  x={x:>3} k={k}  SKIP-> dp[{x+1}][{k}] {old:>8.2f} -> {cur:>8.2f}")

                if k < K:
                    for L in Ls:
                        end = x + L
                        if end > width:
                            continue
                        # in questa posizione, se apro un 
                        # rettangolo [x, x+L), quanti target cells copro dentro e fuori?
                        inside = inside_of(x, end)
                        if inside <= 1e-9:
                            continue
                        outside = L - inside
                        cand = cur + (inside - alpha * outside)
                        # lo score è maggiore di quello che avevo 
                        # prima in dp[end][k+1]? Se sì, aggiorno e registro la scelta.
                        if cand > dp[end][k + 1]:
                            old = dp[end][k + 1]
                            dp[end][k + 1] = cand
                            choice[end][k + 1] = ("take", x, k, L, inside, outside)
                            # print(f"  x={x:>3} k={k}  TAKE-> dp[{end}][{k + 1}] {old:>8.2f} -> {cand:>8.2f}")

        # print("\n--- tabella dp finale (righe=posizione x sull'asse tempo, colonne=k rettangoli usati) ---")
        # header = "x\\k".ljust(6) + "".join(f"k={k:<9}" for k in range(K + 1))
        # print(header)
        # for x in range(width + 1):
        #     vals = "".join(
        #         (f"{dp[x][k]:<11.2f}" if dp[x][k] != NEG else f"{'--':<11}")
        #         for k in range(K + 1)
        #     )
        #     print(f"{x:<6}{vals}")

        best_score, best_x, best_k = NEG, width, 0
        for x in range(width + 1):
            for k in range(K + 1):
                if dp[x][k] > best_score:
                    best_score, best_x, best_k = dp[x][k], x, k

        chosen = []
        x, k = best_x, best_k
        while True:
            step = choice[x][k]
            if step is None:
                break
            if step[0] == "skip":
                _, x, k = step
            else:
                _, start, prev_k, L, inside, outside = step
                chosen.append((start, L, inside, outside))
                x, k = start, prev_k

        chosen.reverse()
        coverage = sum(c[2] for c in chosen)
        return best_score, coverage, chosen

    def max_rectangles_for_row(self, row, L):
        return sum(
            math.ceil((end - start) / L)
            for start, end in self.get_runs(row)
        )

    def min_rectangles_needed(self, grid, L):
        """
        True minimum number of L-wide rectangles needed to fully cover every
        row of the grid, run by run (same logic as max_rectangles_for_row,
        summed across rows). Unlike grid.sum() / L, this accounts for
        fragmentation: a row split into several short runs costs one
        ceil(run_len / L) per run, not one ceil() on the aggregated total.
        """
        return sum(self.max_rectangles_for_row(row, L) for row in grid)
        
    def allocate_shape(self, grid, L, total_rectangles, alpha, continuous_demand):
        rectangles = []

        remaining_rectangles = total_rectangles
       
        for y, row in enumerate(grid):

            if remaining_rectangles <= 0:
                break

            target_mask = 0

            for x, value in enumerate(row):
                if value == 1:
                    target_mask |= 1 << x

            if target_mask == 0:
                continue

            max_by_row = self.max_rectangles_for_row(row, L)
       
            row_budget = min(max_by_row, remaining_rectangles)

            value_row = np.clip(continuous_demand - y, 0.0, 1.0)

            score, coverage, chosen = self.allocate_row_dp_time(
                value_row=value_row,
                Ls=[L],
                max_rectangles=row_budget,
                alpha=alpha,
            )

            for start, L_used, inside, outside in chosen:
                rectangles.append({
                    "x_start": start,
                    "x_end": start + L_used,
                    "y": y,
                    "inside_cells": inside,
                    "outside_cells": outside,
                })

            remaining_rectangles -= len(chosen)
            max_coverage_allowed = coverage

        return rectangles