"""Draw the section 9 result bar charts from the recorded success counts."""

from pathlib import Path

import matplotlib.pyplot as plt

from hurobo.utils.plotting import plot_success_bars

ASSETS = Path(__file__).resolve().parents[1] / "assets"
TEAL, ORANGE, BLUE, GREY = "#1F8A84", "#E06A1F", "#5B9BD5", "#8A96A0"

SCENES = [
    "1 Stock",
    "2 Cleared, LIBERO layout",
    "3 Cleared, new layout",
    "4 Real size, LIBERO layout",
    "5 Real size, new layout",
]
PRETRAINED_BY_SCENE = {
    "lerobot/smolvla_libero": [10, 6, 0, 0, 0],
    "HuggingFaceVLA/smolvla_libero": [8, 3, 0, 0, 0],
}
HELD_OUT = [
    "ours, smolvla_libero init",
    "ours, smolvla_base init",
    "lerobot/smolvla_libero",
    "HuggingFaceVLA/smolvla_libero",
]


def main() -> None:
    """Write results_pretrained_scenes.png and results_held_out.png into assets/."""
    fig = plot_success_bars(
        SCENES,
        PRETRAINED_BY_SCENE,
        trials=10,
        colors={"lerobot/smolvla_libero": TEAL, "HuggingFaceVLA/smolvla_libero": ORANGE},
        xlabel="successes of 10 episodes",
        trials_overrides={("HuggingFaceVLA/smolvla_libero", 3): 8},  # one cell has 8 episodes
    )
    fig.savefig(ASSETS / "results_pretrained_scenes.png", dpi=150)
    plt.close(fig)

    fig = plot_success_bars(
        HELD_OUT,
        {"successes of 20": [14, 3, 0, 0]},
        trials=20,
        colors={"successes of 20": TEAL},
        xlabel="successes of 20 held-out layouts",
    )
    fig.savefig(ASSETS / "results_held_out.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
