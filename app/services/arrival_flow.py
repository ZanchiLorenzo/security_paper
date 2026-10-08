import math
import numpy as np
import torch
import torch.nn as nn
import zuko
from app.services.entities import ArrivalSampler

class Embedder(nn.Module):
    def __init__(self, cardinalities: dict[str, int], embed_dim: int = 8):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cardinalities.values()
        ])

    def forward(self, cats: torch.Tensor) -> torch.Tensor:
        parts = [emb(cats[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)

def load_arrival_flow(path: str) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=False)

    embedder = Embedder(bundle["cardinalities"], embed_dim=bundle["embed_dim"])
    embedder.load_state_dict(bundle["embedder_state"])
    embedder.eval()

    context_dim = bundle["embed_dim"] * len(bundle["cat_cols"])
    flow = zuko.flows.NSF(
        features=1,
        context=context_dim,
        transforms=bundle["transforms"],
        hidden_features=bundle["hidden_features"],
    )
    flow.load_state_dict(bundle["flow_state"])
    flow.eval()

    return {
        "embedder": embedder,
        "flow": flow,
        "code_lookup": bundle["code_lookup"],
        "cat_cols": bundle["cat_cols"],
        "log_adv_mean": bundle["log_adv_mean"],
        "log_adv_std": bundle["log_adv_std"],
        "_sampler_cache": {},
    }

def _encode(model: dict, col: str, value) -> int:
    lookup = model["code_lookup"][col]
    return lookup.get(value, lookup["UNK"])

def _sample_advance(model: dict, car, month, time_band, apt, n_samples: int = 20_000) -> np.ndarray:
    codes = [
        _encode(model, col, val)
        for col, val in zip(model["cat_cols"], [car, month, time_band, apt])
    ]
    cats = torch.tensor([codes] * n_samples, dtype=torch.long)

    with torch.no_grad():
        context = model["embedder"](cats)
        z = model["flow"](context).sample()

    log_adv = z.squeeze(-1) * model["log_adv_std"] + model["log_adv_mean"]
    return torch.exp(log_adv).numpy()

def _flow_to_sampler(model: dict, car, month, time_band, apt,
                      step: float = 1.0, n_samples: int = 20_000, tail_pct: float = 0.1) -> ArrivalSampler:
    
    samples = _sample_advance(model, car, month, time_band, apt, n_samples=n_samples)

    x_min = np.percentile(samples, tail_pct)
    x_max = np.percentile(samples, 100 - tail_pct)

    x = np.arange(math.floor(x_min), math.ceil(x_max) + step, step, dtype=float)

    sorted_s = np.sort(samples)
    cdf = np.searchsorted(sorted_s, x, side="right") / len(sorted_s)
    cdf = np.maximum.accumulate(cdf)
    cdf[0], cdf[-1] = 0.0, 1.0

    return ArrivalSampler(x=x, cdf=cdf)

def get_arrival_sampler(model: dict, car, month, time_band, apt) -> ArrivalSampler:
    key = (car, month, time_band, apt)
    cache = model["_sampler_cache"]

    if key not in cache:
        cache[key] = _flow_to_sampler(model, car, month, time_band, apt)

    return cache[key]