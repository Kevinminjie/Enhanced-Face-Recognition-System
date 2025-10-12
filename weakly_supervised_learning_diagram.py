"""Generate a high quality SVG overview diagram for weakly supervised learning.

The drawing uses layered gradients, soft drop shadows, and elbow connectors to
produce a publication-ready look. Running the script will create the diagram in
``figures/weakly_supervised_learning_overview.svg``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring


# Canvas configuration -----------------------------------------------------
WIDTH, HEIGHT = 2400, 1400
BACKGROUND_GRADIENT = ("#f7faff", "#edf2ff")


# Data models --------------------------------------------------------------


@dataclass
class Node:
    """Represent a labeled rounded rectangle in the diagram."""

    label: str
    column: int
    row: float
    width: int
    height: int
    palette: Tuple[str, str]
    text_color: str = "#1a1c2d"
    shadow: bool = True

    @property
    def position(self) -> Tuple[float, float, float, float]:
        x = COLUMN_X[self.column]
        y = COLUMN_BASE_Y[self.column] + self.row * (ROW_HEIGHT + ROW_SPACING)
        return x, y, self.width, self.height

    def center_right(self) -> Tuple[float, float]:
        x, y, w, h = self.position
        return x + w, y + h / 2

    def center_left(self) -> Tuple[float, float]:
        x, y, _, h = self.position
        return x, y + h / 2


# Layout constants ---------------------------------------------------------
COLUMN_X = [160, 600, 1100, 1640]
COLUMN_BASE_Y = [HEIGHT / 2 - 160, 180, 160, 80]
ROW_HEIGHT = 140
ROW_SPACING = 70


# Diagram construction helpers --------------------------------------------


def prettify_svg(svg_element: Element) -> bytes:
    """Return pretty-printed SVG bytes."""

    return minidom.parseString(tostring(svg_element)).toprettyxml(
        indent="  ", encoding="utf-8"
    )


def create_svg_root() -> Element:
    svg = Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(WIDTH),
            "height": str(HEIGHT),
            "viewBox": f"0 0 {WIDTH} {HEIGHT}",
        },
    )
    add_background(svg)
    add_defs(svg)
    return svg


def add_background(svg: Element) -> None:
    """Add a soft background gradient and subtle vignette."""

    defs = SubElement(svg, "defs")
    gradient = SubElement(
        defs,
        "linearGradient",
        {
            "id": "bg-gradient",
            "x1": "0%",
            "y1": "0%",
            "x2": "0%",
            "y2": "100%",
        },
    )
    SubElement(gradient, "stop", {"offset": "0%", "stop-color": BACKGROUND_GRADIENT[0]})
    SubElement(gradient, "stop", {"offset": "100%", "stop-color": BACKGROUND_GRADIENT[1]})

    SubElement(
        svg,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": str(WIDTH),
            "height": str(HEIGHT),
            "fill": "url(#bg-gradient)",
        },
    )

    # Vignette overlay
    SubElement(
        svg,
        "rect",
        {
            "x": "0",
            "y": "0",
            "width": str(WIDTH),
            "height": str(HEIGHT),
            "fill": "url(#vignette)",
        },
    )


def add_defs(svg: Element) -> None:
    """Add reusable markers, filters, and gradients."""

    defs = next((child for child in svg if child.tag == "defs"), None)
    if defs is None:
        defs = SubElement(svg, "defs")

    # Soft vignette using radial gradient
    vignette = SubElement(
        defs,
        "radialGradient",
        {
            "id": "vignette",
            "cx": "50%",
            "cy": "40%",
            "r": "75%",
        },
    )
    SubElement(
        vignette,
        "stop",
        {"offset": "60%", "stop-color": "#ffffff", "stop-opacity": "0"},
    )
    SubElement(
        vignette,
        "stop",
        {"offset": "100%", "stop-color": "#000000", "stop-opacity": "0.08"},
    )

    # Drop shadow filter for nodes
    shadow = SubElement(
        defs,
        "filter",
        {
            "id": "soft-shadow",
            "x": "-20%",
            "y": "-20%",
            "width": "160%",
            "height": "160%",
            "filterUnits": "objectBoundingBox",
        },
    )
    SubElement(shadow, "feGaussianBlur", {"in": "SourceAlpha", "stdDeviation": "12"})
    SubElement(shadow, "feOffset", {"dx": "0", "dy": "16", "result": "offsetblur"})
    merge = SubElement(shadow, "feMerge")
    SubElement(merge, "feMergeNode", {"in": "offsetblur"})
    SubElement(merge, "feMergeNode", {"in": "SourceGraphic"})

    # Arrow marker for elbow connectors
    marker = SubElement(
        defs,
        "marker",
        {
            "id": "arrow",
            "viewBox": "0 0 10 10",
            "refX": "9.5",
            "refY": "5",
            "markerWidth": "14",
            "markerHeight": "14",
            "orient": "auto",
        },
    )
    SubElement(marker, "path", {"d": "M 0 0 L 10 5 L 0 10 z", "fill": "#354369"})


def define_gradient(svg: Element, gradient_id: str, start: str, end: str) -> None:
    defs = next((child for child in svg if child.tag == "defs"), None)
    if defs is None:
        defs = SubElement(svg, "defs")

    gradient = SubElement(
        defs,
        "linearGradient",
        {
            "id": gradient_id,
            "x1": "0%",
            "y1": "0%",
            "x2": "100%",
            "y2": "100%",
        },
    )
    SubElement(gradient, "stop", {"offset": "0%", "stop-color": start})
    SubElement(gradient, "stop", {"offset": "100%", "stop-color": end})


def wrap_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    lines: List[str] = []
    current = ""
    for char in text:
        current += char
        if len(current) >= max_chars:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines


def add_box(svg: Element, node: Node) -> None:
    x, y, w, h = node.position
    gradient_id = f"grad-{hash(node.label) & 0xFFFF:x}"
    define_gradient(svg, gradient_id, *node.palette)

    group = SubElement(svg, "g", {"filter": "url(#soft-shadow)"}) if node.shadow else svg

    SubElement(
        group,
        "rect",
        {
            "x": f"{x}",
            "y": f"{y}",
            "width": f"{w}",
            "height": f"{h}",
            "rx": "30",
            "ry": "30",
            "fill": f"url(#{gradient_id})",
            "stroke": "#1f2a44",
            "stroke-width": "3",
        },
    )

    lines = wrap_text(node.label, 8 if w < 320 else 10)
    for i, line in enumerate(lines):
        SubElement(
            group,
            "text",
            {
                "x": f"{x + w / 2}",
                "y": f"{y + h / 2 - (len(lines) - 1) * 24 + i * 48}",
                "fill": node.text_color,
                "font-size": "40",
                "font-family": '"Source Han Sans","PingFang SC","Noto Sans SC",sans-serif',
                "font-weight": "600",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
            },
        ).text = line


def add_elbow(svg: Element, start: Tuple[float, float], end: Tuple[float, float], offset: int = 120) -> None:
    """Draw an elbow connector with rounded corner effect."""

    (x1, y1), (x2, y2) = start, end
    mid_x = x1 + offset
    mid_y = y2
    path = f"M {x1} {y1} L {mid_x} {y1} Q {mid_x + 10} {y1} {mid_x + 10} {y1 + 10} L {mid_x + 10} {mid_y - 10} Q {mid_x + 10} {mid_y} {mid_x + 20} {mid_y} L {x2} {y2}"

    SubElement(
        svg,
        "path",
        {
            "d": path,
            "fill": "none",
            "stroke": "#354369",
            "stroke-width": "4",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "marker-end": "url(#arrow)",
        },
    )


def add_title(svg: Element) -> None:
    SubElement(
        svg,
        "text",
        {
            "x": str(WIDTH / 2),
            "y": "110",
            "fill": "#1f2a44",
            "font-size": "64",
            "font-weight": "700",
            "font-family": '"Source Han Sans","PingFang SC","Noto Sans SC",sans-serif',
            "text-anchor": "middle",
        },
    ).text = "弱监督学习方法全景图"

    SubElement(
        svg,
        "text",
        {
            "x": str(WIDTH / 2),
            "y": "180",
            "fill": "#46506c",
            "font-size": "34",
            "font-family": '"Source Han Sans","PingFang SC","Noto Sans SC",sans-serif',
            "text-anchor": "middle",
        },
    ).text = "分层结构、非重叠布局与折线箭头"


def connect(svg: Element, parent: Node, child: Node, offset: int = 120) -> None:
    add_elbow(svg, parent.center_right(), child.center_left(), offset)


def build_diagram(svg: Element) -> None:
    add_title(svg)

    # Define nodes ---------------------------------------------------------
    root = Node("弱监督学习", 0, 0.0, 340, 200, ("#7b9bff", "#5b6cea"), "#ffffff")

    tier_nodes = [
        Node("数据增广", 1, 0.0, 360, 170, ("#39d98a", "#1aa66b"), "#ffffff"),
        Node("迁移学习", 1, 1.3, 360, 170, ("#b08dff", "#8052ff"), "#ffffff"),
        Node("交互式分割", 1, 2.6, 360, 170, ("#ffa94d", "#ff7a1f"), "#ffffff"),
    ]

    augmentation = [
        Node("传统增广", 2, -0.1, 380, 150, ("#d7fbe8", "#9cebd1")),
        Node("合成增广", 2, 0.9, 380, 150, ("#d7fbe8", "#9cebd1")),
    ]

    transfer = [
        Node("预训练模型", 2, 2.3, 380, 150, ("#ede2ff", "#c5b5ff")),
        Node("域适应", 2, 3.3, 380, 150, ("#ede2ff", "#c5b5ff")),
    ]

    interactive = [
        Node("DeepIGeos", 2, 4.4, 380, 150, ("#ffe6d0", "#ffbf80")),
        Node("BIFSeg", 2, 5.4, 380, 150, ("#ffe6d0", "#ffbf80")),
        Node("GM Interacting", 2, 6.4, 380, 150, ("#ffe6d0", "#ffbf80")),
    ]

    traditional_leaves = [
        Node("改变图像噪声", 3, -1.0, 420, 140, ("#f0fdf4", "#d1fae5")),
        Node("弹性形变", 3, -0.1, 420, 140, ("#f0fdf4", "#d1fae5")),
        Node("颜色变换", 3, 0.8, 420, 140, ("#f0fdf4", "#d1fae5")),
        Node("裁剪/翻转", 3, 1.7, 420, 140, ("#f0fdf4", "#d1fae5")),
    ]

    synthetic_leaves = [
        Node("条件GAN", 3, 3.2, 420, 140, ("#f0fdf4", "#d1fae5")),
    ]

    # Render nodes ---------------------------------------------------------
    for node in [root, *tier_nodes, *augmentation, *transfer, *interactive, *traditional_leaves, *synthetic_leaves]:
        add_box(svg, node)

    # Connections ----------------------------------------------------------
    for tier_node in tier_nodes:
        connect(svg, root, tier_node, offset=180)

    for aug_node in augmentation:
        connect(svg, tier_nodes[0], aug_node)

    for leaf in traditional_leaves:
        connect(svg, augmentation[0], leaf, offset=140)

    for leaf in synthetic_leaves:
        connect(svg, augmentation[1], leaf, offset=140)

    for transfer_node in transfer:
        connect(svg, tier_nodes[1], transfer_node)

    for interactive_node in interactive:
        connect(svg, tier_nodes[2], interactive_node)


def main() -> Path:
    svg = create_svg_root()
    build_diagram(svg)

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "weakly_supervised_learning_overview.svg"

    with output_path.open("wb") as fh:
        fh.write(prettify_svg(svg))

    return output_path


if __name__ == "__main__":
    path = main()
    print(f"SVG exported to: {path}")

