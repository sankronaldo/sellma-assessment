"""Build report/report.pdf from RESEARCH_NOTE.md using fpdf2."""

import pathlib
import re
from fpdf import FPDF

ROOT = pathlib.Path(__file__).parent.parent
SRC = ROOT / "RESEARCH_NOTE.md"
OUT = ROOT / "report" / "report.pdf"


def clean(s: str) -> str:
    return (s
        .replace("–", "-").replace("—", "-")
        .replace("’", "'").replace("“", '"').replace("”", '"')
        .replace("→", "->").replace("←", "<-").replace("•", "*")
        .replace("≥", ">=").replace("≤", "<=")
        .replace("±", "+/-").replace("×", "x")
        .replace("ä", "a").replace("ö", "o").replace("å", "a")
        .replace("Ä", "A").replace("Ö", "O").replace("Å", "A")
        .replace("é", "e").replace("ø", "o").replace("æ", "ae")
        .replace("​", "")
        .replace("‖", "||")  # double vertical line (norm notation)
        .replace("−", "-")   # minus sign
        .replace("·", "*")   # middle dot
        .replace("…", "...")  # ellipsis
        .replace("→", "->").replace("⇒", "=>")
        .replace("α", "alpha").replace("β", "beta")
        .replace("χ", "chi").replace("²", "^2")
        .replace("÷", "/")
    )


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def render_inline(pdf: PDF, line: str, size: int = 10) -> None:
    """Write a line with inline bold/italic handled."""
    parts = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", line)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font("Helvetica", "B", size)
            pdf.write(5.5, clean(part[2:-2]))
            pdf.set_font("Helvetica", "", size)
        elif part.startswith("*") and part.endswith("*"):
            pdf.set_font("Helvetica", "I", size)
            pdf.write(5.5, clean(part[1:-1]))
            pdf.set_font("Helvetica", "", size)
        else:
            pdf.write(5.5, clean(part))


def flush_table(pdf: PDF, rows: list) -> None:
    if not rows:
        return
    pdf.ln(2)
    n_cols = max(len(r) for r in rows)
    page_w = 166.0
    col_w = page_w / n_cols

    for ri, row in enumerate(rows):
        x0 = pdf.get_x()
        y0 = pdf.get_y()
        is_header = ri == 0
        for ci in range(n_cols):
            cell = row[ci] if ci < len(row) else ""
            pdf.set_font("Helvetica", "B" if is_header else "", 8)
            fill_color = (215, 225, 245) if is_header else ((248, 248, 252) if ri % 2 == 0 else (255, 255, 255))
            pdf.set_fill_color(*fill_color)
            pdf.set_xy(x0 + ci * col_w, y0)
            # strip inline markdown from table cells
            cell_clean = re.sub(r"\*+([^*]+)\*+", r"\1", clean(cell))
            pdf.multi_cell(col_w, 5, cell_clean, border=1, fill=True)
        row_h = pdf.get_y() - y0
        pdf.set_xy(x0, y0 + row_h)
    pdf.ln(3)


def build(src_path: pathlib.Path, out_path: pathlib.Path) -> None:
    text = src_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(22, 20, 22)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    table_rows: list = []
    in_code = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Flush table when we hit a non-table line
        if table_rows and not line.startswith("|"):
            flush_table(pdf, table_rows)
            table_rows = []

        # Horizontal rule
        if line.strip() in ("---", "***", "___"):
            pdf.set_draw_color(180)
            pdf.line(22, pdf.get_y(), 188, pdf.get_y())
            pdf.ln(3)
            i += 1
            continue

        # H1
        if line.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(15, 35, 90)
            pdf.multi_cell(0, 9, clean(line[2:]))
            pdf.set_text_color(0)
            pdf.ln(2)
            i += 1
            continue

        # H2
        if line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(30, 50, 120)
            pdf.multi_cell(0, 7, clean(line[3:]))
            pdf.set_text_color(0)
            pdf.ln(1)
            i += 1
            continue

        # H3
        if line.startswith("### "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 6, clean(line[4:]))
            pdf.set_text_color(0)
            i += 1
            continue

        # Table row
        if line.startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not all(set(c) <= set("-: ") for c in cells):
                table_rows.append(cells)
            i += 1
            continue

        # Code fence
        if line.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            pdf.set_font("Courier", "", 8)
            pdf.set_fill_color(242, 242, 242)
            pdf.set_x(26)
            pdf.cell(0, 5, clean(line), new_x="LMARGIN", new_y="NEXT", fill=True)
            i += 1
            continue

        # Bullet point
        if line.startswith("- ") or line.startswith("* "):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(27)
            pdf.write(5.5, "-  ")
            render_inline(pdf, line[2:])
            pdf.ln()
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\d+)\.\s+(.+)", line)
        if m:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(27)
            pdf.write(5.5, f"{m.group(1)}.  ")
            render_inline(pdf, m.group(2))
            pdf.ln()
            i += 1
            continue

        # Normal text
        if line.strip():
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(22)
            render_inline(pdf, line)
            pdf.ln()
        else:
            pdf.ln(3)

        i += 1

    # Flush any remaining table
    if table_rows:
        flush_table(pdf, table_rows)

    out_path.parent.mkdir(exist_ok=True)
    pdf.output(str(out_path))
    print(f"PDF written: {out_path} ({pdf.page} pages)")


if __name__ == "__main__":
    build(SRC, OUT)
