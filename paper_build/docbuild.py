"""Helpers that fill the official IEEE conference template (converted to transitional OOXML)."""
import copy
import re

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class Paper:
    def __init__(self, tpl):
        self.doc = Document(tpl)
        body = self.doc.element.body
        els = list(body)
        self.title_p, self.author_p, self.anchor = els[0], els[3], els[98]
        self.tbl_proto = copy.deepcopy(els[75])
        # drop template guidance: note under title, extra author rows, all sample body text
        for i, e in enumerate(els):
            if i in (1,) or 4 <= i <= 97:
                body.remove(e)
        for r in list(self.anchor):
            if r.tag != W + "pPr":
                self.anchor.remove(r)
        ppr = self.anchor.find(W + "pPr")
        for tag in ("pStyle", "numPr", "rPr"):
            e = ppr.find(W + tag)
            if e is not None:
                ppr.remove(e)

    # ---------------------------------------------------------------- runs with *italic* / **bold** / ^sup^ / ~sub~
    @staticmethod
    def add_runs(p, text, size=None, base_italic=False):
        tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~)", text)
        for t in tokens:
            if not t:
                continue
            b = i = False
            va = None
            if t.startswith("**"):
                t, b = t[2:-2], True
            elif t.startswith("*"):
                t, i = t[1:-1], True
            elif t.startswith("^"):
                t, va = t[1:-1], "superscript"
            elif t.startswith("~"):
                t, va = t[1:-1], "subscript"
            r = p.add_run(t)
            if b:
                r.bold = True
            if i or base_italic:
                r.italic = True
            if va == "superscript":
                r.font.superscript = True
            elif va == "subscript":
                r.font.subscript = True
            if size:
                r.font.size = Pt(size)
        return p

    def _new_p(self, style):
        p = self.doc.add_paragraph(style=style)
        self.anchor.addprevious(p._p)
        return p

    def title(self, text):
        for r in list(self.title_p):
            if r.tag != W + "pPr":
                self.title_p.remove(r)
        r = OxmlElement("w:r"); t = OxmlElement("w:t"); t.text = text; r.append(t); self.title_p.append(r)

    def authors(self, blocks):
        """blocks: list of lists of lines; line 2/3 italic like the template."""
        p = self.author_p
        for r in list(p):
            if r.tag != W + "pPr":
                p.remove(r)
        cols = p.find(f".//{W}cols")
        cols.set(W + "num", str(len(blocks)))
        from docx.text.paragraph import Paragraph
        para = Paragraph(p, self.doc._body)
        for bi, lines in enumerate(blocks):
            for li, line in enumerate(lines):
                r = para.add_run(line)
                r.font.size = Pt(9)
                if li in (1, 2):
                    r.italic = True
                if li < len(lines) - 1:
                    r.add_break()
            if bi < len(blocks) - 1:
                para.add_run().add_break(WD_BREAK.COLUMN)

    def abstract(self, text):
        p = self._new_p("Abstract")
        r = p.add_run("Abstract"); r.italic = True
        p.add_run("—")
        self.add_runs(p, text)

    def keywords(self, text):
        p = self._new_p("Keywords")
        r = p.add_run("Keywords—"); r.italic = True; r.bold = True
        r2 = p.add_run(text); r2.italic = True; r2.bold = True

    def h1(self, text):
        return self.add_runs(self._new_p("Heading 1"), text)

    def h2(self, text):
        return self.add_runs(self._new_p("Heading 2"), text)

    def para(self, text, style="Body Text"):
        return self.add_runs(self._new_p(style), text)

    def bullet(self, text):
        return self.add_runs(self._new_p("bullet list"), text)

    @staticmethod
    def _fix_font(run):
        # the template's "equation" style forces Symbol font (ascii a-z/A-Z render
        # as Greek glyphs in that font); override back to a normal text font so
        # plain Latin letters and real Unicode math symbols both render correctly.
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.find(qn("w:rFonts"))
        if rFonts is None:
            rFonts = OxmlElement("w:rFonts")
            rPr.append(rFonts)
        for attr in ("w:ascii", "w:hAnsi", "w:cs"):
            rFonts.set(qn(attr), "Times New Roman")

    def equation(self, expr, num):
        p = self._new_p("equation")
        r0 = p.add_run("\t")
        self._fix_font(r0)
        self.add_runs(p, expr)
        rN = p.add_run(f"\t({num})")
        self._fix_font(rN)
        for r in p.runs:
            self._fix_font(r)
        return p

    def figure(self, path, caption, width=3.3):
        p = self._new_p("Normal")
        p.alignment = 1
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Inches(width))
        self.add_runs(self._new_p("figure caption"), caption)

    def table(self, caption, header, rows, widths, bold_rows=(), note=None, font=7.5):
        self.add_runs(self._new_p("table head"), caption)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        tblPr = t._tbl.tblPr
        proto = self.tbl_proto.find(W + "tblPr")
        t._tbl.remove(tblPr)
        t._tbl.insert(0, copy.deepcopy(proto))
        for ci, w in enumerate(widths):
            for row in t.rows:
                row.cells[ci].width = Inches(w)
        grid = t._tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w * 1440)))
        for ri, vals in enumerate([header] + rows):
            for ci, v in enumerate(vals):
                cell = t.rows[ri].cells[ci]
                p = cell.paragraphs[0]
                p.style = self.doc.styles["table col head" if ri == 0 else "table copy"]
                p.alignment = 1 if (ri == 0 or ci > 0) else 0
                self.add_runs(p, str(v), size=font)
                if ri == 0 or (ri - 1) in bold_rows:
                    for r in p.runs:
                        r.bold = True
        self.anchor.addprevious(t._tbl)
        if note:
            p = self._new_p("Normal")
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(6)
            self.add_runs(p, note, size=7)
        else:
            sp = self._new_p("Normal"); sp.paragraph_format.space_after = Pt(4)

    def references(self, refs):
        for r in refs:
            self.add_runs(self._new_p("references"), r)

    def save(self, path):
        self.doc.save(path)
