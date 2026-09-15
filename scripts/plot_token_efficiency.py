"""Rebuild the total-token figure using the paper's vector-figure style.

Accuracy corrections follow EdgeIG/main.tex, Table 1. Other accuracy values
and total-token counts were recovered from the original committed figure's
vector coordinates (token counts rounded to the nearest token), because its
underlying plotting script and measurement records were not supplied.

Style reference: EdgeIG/figures/accuracy_vs_prompt_tokens_iclr.pdf.
The same diameter-per-token scale is used in both panels, with the largest
bubble matching the reference's 50.4-point diameter. No minimum-size offset.
Run with Python and reportlab; the output PDF remains vector artwork.
"""

from pathlib import Path

from reportlab import rl_config
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "accuracy_vs_total_tokens_all.pdf"
METHODS = ["Complete Graph", "Random Graph", "AgentPrune", "ARG-Designer", "EdgeIG"]
COLORS = ["#5ca1c5", "#8dcdb6", "#ffe39a", "#b9ddb0", "#ffbc77"]
# (accuracy percent, total inference tokens = prompt + completion)
DATA = {
    "MMLU": [(77.12418301, 2710575), (74.50980392, 2071519),
             (79.95642702, 1746388), (76.68, 685788), (80.39, 1302399)],
    "AQuA": [(84.43396226, 1721473), (83.80503145, 1570206),
             (84.11949686, 1466862), (83.95, 1040908), (86.79, 1233637)],
}
MAX_TOKENS = max(tokens for points in DATA.values() for _, tokens in points)
MAX_RADIUS_PT = 25.2
TEXT = HexColor("#222222")


def text(c, x, y, value, size, font="Times-Roman", align="center"):
    c.setFillColor(TEXT)
    c.setFont(font, size)
    draw = {"center": c.drawCentredString, "right": c.drawRightString,
            "left": c.drawString}[align]
    draw(x, y, value)


def panel(c, name, x0, xlim, ylim, xticks, yticks, labels):
    y0, width, height = 30, 153, 148

    def px(value):
        return x0 + width * (value - xlim[0]) / (xlim[1] - xlim[0])

    def py(value):
        return y0 + height * (value - ylim[0]) / (ylim[1] - ylim[0])

    c.setFillColor(HexColor("#ffffff"))
    c.rect(x0, y0, width, height, fill=1, stroke=0)
    c.saveState()
    c.setStrokeColor(HexColor("#dadada"))
    c.setLineWidth(.3)
    c.setDash(1, 1)
    for value in xticks:
        c.line(px(value), y0, px(value), y0 + height)
    for value in yticks:
        c.line(x0, py(value), x0 + width, py(value))
    c.restoreState()
    c.setStrokeColor(TEXT)
    c.setLineWidth(.6)
    c.rect(x0, y0, width, height, fill=0, stroke=1)
    for value in xticks:
        text(c, px(value), 20, str(value), 7.3)
    for value in yticks:
        text(c, x0 - 5, py(value) - 2.5, f"{value:.1f}", 7.3, align="right")
    text(c, x0 + width / 2, 6, "Accuracy (%)", 9.5)
    c.saveState()
    c.translate(x0 - 24, y0 + height / 2)
    c.rotate(90)
    text(c, 0, 0, "token consumption", 8.7)
    c.restoreState()

    for (accuracy, tokens), color in zip(DATA[name], COLORS):
        x, y, radius = px(accuracy), py(tokens / 1e6), MAX_RADIUS_PT * tokens / MAX_TOKENS
        assert x - radius > x0 and x + radius < x0 + width
        assert y - radius > y0 and y + radius < y0 + height
        c.setFillColor(HexColor(color))
        c.setStrokeColor(HexColor("#919191"))
        c.setLineWidth(.65)
        c.circle(x, y, radius, fill=1, stroke=1)
        c.setStrokeColor(TEXT)
        c.setLineWidth(.55)
        c.line(x - 1.9, y, x + 1.9, y)
        c.line(x, y - 1.9, x, y + 1.9)

    title_x = x0 + .3 if name == "MMLU" else x0 + width - 48
    c.setFillColor(HexColor("#e1eed7" if name == "MMLU" else "#dfecf6"))
    c.rect(title_x, 157, 47.7, 20.7, fill=1, stroke=0)
    text(c, title_x + 23.85, 164, name, 11, "Times-Bold")
    for method, (_, tokens), (x, y) in zip(METHODS, DATA[name], labels):
        text(c, x, y, method, 6.6)
        text(c, x, y - 7, f"({tokens:.2e} tokens)", 6.2)


def main():
    # Binary Flate streams also keep Git from treating a PDF as CRLF text.
    rl_config.useA85 = 0
    c = canvas.Canvas(str(OUTPUT), pagesize=(396, 188), invariant=1)
    c.setTitle("Accuracy and total-token consumption on MMLU and AQuA")
    c.setSubject("Inference prompt plus completion tokens; bubble diameter proportional to tokens.")
    panel(c, "MMLU", 35, (73, 82), (0, 3.65), [74, 76, 78, 80, 82],
          [.5, 1, 1.5, 2, 2.5, 3, 3.5],
          [(148, 169), (76, 85), (153, 131), (98, 44), (151, 54)])
    panel(c, "AQuA", 237, (83, 87.6), (.75, 2.0), [83, 84, 85, 86, 87],
          [1, 1.2, 1.4, 1.6, 1.8],
          [(281, 171), (328, 136), (304, 93), (273, 45), (362, 113)])
    c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
