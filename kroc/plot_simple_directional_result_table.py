"""Render the simple Post-DOB/model-only comparison as a presentation table."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


OUTPUT = Path(__file__).resolve().parent / "figures" / (
    "2026-11-09_simple_directional_results_table.png"
)
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def add_table(axis, title, columns, rows):
    axis.axis("off")
    axis.set_title(
        title, fontsize=18, fontweight="bold", loc="left", pad=13,
        fontproperties=FONT,
    )
    table = axis.table(
        cellText=rows, colLabels=columns, cellLoc="center", colLoc="center",
        loc="center", bbox=[0.0, 0.03, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    for (row, column), cell in table.get_celld().items():
        cell.get_text().set_fontproperties(FONT)
        cell.set_edgecolor("#d0d7de")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1f4e78")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row in (1, 4):
            cell.set_facecolor("#f6f8fa")
        elif row in (2, 5):
            cell.set_facecolor("#e8f4fd")
        else:
            cell.set_facecolor("#fff4dd")
        if column in (0, 1) and row > 0:
            cell.get_text().set_fontweight("bold")
    return table


def main():
    plt.rcParams.update({"axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 10.3))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "센서 노이즈 환경: Post-DOB와 Model-only의 상태 제약 비교",
        fontsize=23, fontweight="bold", y=0.985, fontproperties=FONT,
    )
    fig.text(
        0.5, 0.942,
        "5 paired seeds · headroom 제외 · 동적 마진 m0=0, Tr=0.1 s, 최대 10%",
        ha="center", fontsize=12.5, color="#4b5563",
        fontproperties=FONT,
    )

    position_rows = [
        ["Post-DOB", "기본 soft 0%", "4.273 ± 1.491", "21.4 ± 3.9", "0.5811 ± 0.1299"],
        ["Post-DOB", "동적 0–10%", "4.897 ± 3.046", "1.2 ± 1.8", "0.0129 ± 0.0222"],
        ["Post-DOB", "고정 10%", "2.615 ± 0.394", "0", "0"],
        ["Model-only", "기본 soft 0%", "145.746 ± 80.012", "68.4 ± 4.4", "3.1821 ± 0.5469"],
        ["Model-only", "동적 0–10%", "130.828 ± 51.176", "65.6 ± 6.2", "2.9411 ± 0.5609"],
        ["Model-only", "고정 10%", "105.071 ± 49.320", "40.0 ± 9.8", "1.3556 ± 0.3972"],
    ]
    add_table(
        axes[0], "A. 위치 상한 외란 시나리오",
        ["DOB 구조", "제약 방식", "추종 RMSE [mm]", "위치 위반 [step]", "최대 위치 초과 [rad]"],
        position_rows,
    )

    velocity_rows = [
        ["Post-DOB", "기본 soft 0%", "2.010 ± 0.296", "1.4 ± 1.9", "0.1444 ± 0.1990"],
        ["Post-DOB", "동적 0–10%", "1.914 ± 0.148", "0.4 ± 0.9", "0.0140 ± 0.0313"],
        ["Post-DOB", "고정 10%", "1.753 ± 0.315", "0", "0"],
        ["Model-only", "기본 soft 0%", "12.641 ± 5.268", "8.2 ± 0.4", "1.5450 ± 0.3495"],
        ["Model-only", "동적 0–10%", "12.373 ± 7.416", "8.0 ± 0.7", "1.4610 ± 0.4061"],
        ["Model-only", "고정 10%", "16.062 ± 9.270", "7.0 ± 0.0", "1.0429 ± 0.3366"],
    ]
    add_table(
        axes[1], "B. 속도 상한 외란 시나리오",
        ["DOB 구조", "제약 방식", "추종 RMSE [mm]", "속도 위반 [step]", "최대 속도 초과 [rad/s]"],
        velocity_rows,
    )
    fig.text(
        0.02, 0.012,
        "색상: 파랑=동적 마진, 주황=고정 10%, 회색=기본 soft · 값은 mean ± sample standard deviation",
        fontsize=10.5, color="#4b5563", fontproperties=FONT,
    )
    fig.subplots_adjust(top=0.895, bottom=0.045, left=0.025, right=0.975, hspace=0.22)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
