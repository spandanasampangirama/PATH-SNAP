"""Assemble technical report markdown sections into a consolidated report and HTML.  Owner: P4"""
import os
from pathlib import Path
import markdown

report_dir = Path(__file__).resolve().parent
sections_dir = report_dir / "sections"

section_files = sorted(sections_dir.glob("*.md"))
print(f"Found {len(section_files)} report section(s):")
for f in section_files:
    print(f"  - {f.name}")

combined_md = """# PathSnap: Distributed Map-Matching & Route Reconstruction
**Technical Report & System Architecture**  
**Course / Event:** UE24CS352A Machine Learning Hackathon, PES University  
**Date:** November 2026  

---

"""

for f in section_files:
    content = f.read_text(encoding="utf-8")
    combined_md += f"\n\n{content}\n\n---\n"

# Output consolidated markdown
out_md = report_dir / "PathSnap_Technical_Report.md"
out_md.write_text(combined_md, encoding="utf-8")
print(f"Compiled consolidated markdown to: {out_md}")

# Convert to styled HTML document
html_body = markdown.markdown(combined_md, extensions=["tables", "fenced_code", "toc"])

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PathSnap — Technical Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.6;
        color: #1a202c;
        max-width: 900px;
        margin: 40px auto;
        padding: 0 20px;
    }}
    h1, h2, h3, h4 {{
        color: #2d3748;
        font-weight: 700;
        margin-top: 1.5em;
    }}
    h1 {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3em; }}
    h2 {{ border-bottom: 1px solid #edf2f7; padding-bottom: 0.2em; }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        font-size: 0.95em;
    }}
    th, td {{
        border: 1px solid #cbd5e0;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f7fafc;
        font-weight: 600;
    }}
    tr:nth-child(even) {{ background-color: #f8fafc; }}
    code {{
        background-color: #edf2f7;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: Consolas, Monaco, monospace;
        font-size: 0.9em;
    }}
    pre {{
        background-color: #2d3748;
        color: #f7fafc;
        padding: 15px;
        border-radius: 6px;
        overflow-x: auto;
    }}
    pre code {{
        background-color: transparent;
        color: inherit;
        padding: 0;
    }}
    hr {{
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 30px 0;
    }}
    @media print {{
        body {{ max-width: 100%; margin: 20px; }}
        pre, table {{ page-break-inside: avoid; }}
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

out_html = report_dir / "PathSnap_Technical_Report.html"
out_html.write_text(html_doc, encoding="utf-8")
print(f"Compiled styled HTML report to: {out_html}")
