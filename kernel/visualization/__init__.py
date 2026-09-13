"""每個 stage 一張疊圖，供人工查驗。

    output/
    ├── 000000000285_1_masks.jpg    輪廓疊色 + bbox + 物種標籤
    ├── 000000000285_2_eyes.jpg     疊在 1 之上，加上眼睛位置與分數
    ├── 000000000285_3_depth.jpg    深度偽彩圖 + 輪廓 + 取樣視窗
    └── 000000000285_4_measure.jpg  兩種距離的連線與數字

用法::

    from kernel.visualization import render_scene
    render_scene(image, scene, Path("output"))

沒跑到的 stage 不會產生檔案（renderer 回傳 None），不是報錯。
"""

from kernel.visualization.draw import (
    ALL_STAGES,
    DIM_FACTOR,
    STAGE_FILENAMES,
    LabelPlacer,
    draw_depth,
    draw_eyes,
    draw_masks,
    draw_measurements,
    render_scene,
    save,
)
from kernel.visualization.palette import (
    CROSS_OBJECT_COLOR,
    FAILURE_COLOR,
    SAMPLE_WINDOW_COLOR,
    color_for,
)

__all__ = [
    "ALL_STAGES",
    "CROSS_OBJECT_COLOR",
    "DIM_FACTOR",
    "FAILURE_COLOR",
    "SAMPLE_WINDOW_COLOR",
    "STAGE_FILENAMES",
    "LabelPlacer",
    "color_for",
    "draw_depth",
    "draw_eyes",
    "draw_masks",
    "draw_measurements",
    "render_scene",
    "save",
]
