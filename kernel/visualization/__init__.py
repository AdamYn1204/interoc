"""每個 stage 一張疊圖，供人工查驗。

    output/
    ├── 000000000285_1_masks.jpg        輪廓疊色 + bbox + 物種標籤
    ├── 000000000285_2_eyes.jpg         疊在 1 之上，加上眼睛位置與分數
    └── 000000000285_3_interocular.jpg  疊在 2 之上，加上雙眼連線與像素距離

用法::

    from kernel.visualization import render_scene
    render_scene(image, scene, Path("output"))

之後的 stage 各自再加一張，疊在前一張之上。
"""

from kernel.visualization.draw import (
    ALL_STAGES,
    STAGE_FILENAMES,
    LabelPlacer,
    draw_eyes,
    draw_interocular,
    draw_masks,
    render_scene,
    save,
)
from kernel.visualization.palette import MEASUREMENT_COLOR, color_for

__all__ = [
    "ALL_STAGES",
    "MEASUREMENT_COLOR",
    "STAGE_FILENAMES",
    "LabelPlacer",
    "color_for",
    "draw_eyes",
    "draw_interocular",
    "draw_masks",
    "render_scene",
    "save",
]
