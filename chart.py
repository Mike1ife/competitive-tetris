import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _parse_model_name(model_name: str):
    STRATEGY_MAP = {"neu": "NE", "off": "OF", "def": "DF"}
    OPPONENT_MAP = {"rnd": "RA", "agent": "AG", "heu": "HE", "hyb": "HY"}
    VERSION_MAP = {"v1": "1", "v2": "2"}
    strategy, _, opponent, version = model_name.split("_")
    return STRATEGY_MAP[strategy] + OPPONENT_MAP[opponent] + VERSION_MAP[version]


def _sort_df(df):
    strategy_order = ["off", "def", "neu"]
    opponent_order = ["rnd", "agent", "heu", "hyb"]

    strategy_cat = pd.Categorical(
        df["player"].str.extract(r"^(off|def|neu)")[0],
        categories=strategy_order,
        ordered=True,
    )
    opponent_cat = pd.Categorical(
        df["player"].str.extract(r"vs_(rnd|agent|heu|hyb)")[0],
        categories=opponent_order,
        ordered=True,
    )

    version_cat = df["player"].str.extract(r"_v(\d+)$")[0].astype(int)

    return (
        df.assign(_strategy=strategy_cat, _opponent=opponent_cat, _version=version_cat)
        .sort_values(["_strategy", "_opponent", "_version"])
        .drop(columns=["_strategy", "_opponent", "_version"])
    )


df = pd.read_csv("res/first_round_tournament.csv")
df = _sort_df(df)

models = [_parse_model_name(m) for m in df["player"].tolist()]
win_rates = df["win_pct"].tolist()
colors = ["#E24B4A"] * 8 + ["#1D9E75"] * 8 + ["#378ADD"] * 8

fig, ax = plt.subplots(figsize=(10, 11))

bars = ax.barh(models, win_rates, color=colors, height=0.65)

ax.set_xlabel("Win Rate (%)", fontsize=12)
ax.set_xlim(0, 70)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=11)
ax.grid(axis="x", color="gray", alpha=0.2, linewidth=0.8)
ax.set_axisbelow(True)
ax.invert_yaxis()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_patches = [
    mpatches.Patch(color=colors[0], label="Offensive"),
    mpatches.Patch(color=colors[8], label="Defensive"),
    mpatches.Patch(color=colors[16], label="Neutral"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=11, framealpha=0.8)

plt.tight_layout()
plt.savefig("res/first_round_winrate.png", dpi=1000, bbox_inches="tight")
plt.show()
