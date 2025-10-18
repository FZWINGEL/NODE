#!/usr/bin/env python3
"""
transcribe.py

Usage:
  python transcribe.py ACLA.pdf
  python transcribe.py ACLA.pdf --outdir out_acla
  python transcribe.py ACLA.pdf --charts

What it does:
- Runs PaddleOCR PP-StructureV3 on the input PDF
- Writes per-page Markdown + assets into a subfolder
- Combines all page_XXXX.md files into a single 'document.md' with image links preserved

Requirements (same environment you already used):
  pip install -U "paddleocr[doc-parser]" "paddlepaddle>=3.0.0"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Import the pipeline
try:
    from paddleocr import PPStructureV3
except Exception as e:
    print("ERROR: Could not import paddleocr.PPStructureV3. "
          "Install with: pip install -U 'paddleocr[doc-parser]' 'paddlepaddle>=3.0.0'", file=sys.stderr)
    raise

def run_ppstruct(pdf_path: Path, outdir: Path, use_charts: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Create the pipeline (tuned for stability on Windows)
    pipe = PPStructureV3(
        # You can flip this to True if you specifically want unwarping
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_chart_recognition=bool(use_charts),
        # The defaults for table/formula/layout are fine; you can expose these if needed:
        # use_table_recognition=True,
        # use_formula_recognition=True,
        # use_region_detection=True,
        # use_doc_preprocessor=True,
    )

    # Run inference; PP-StructureV3 handles PDFs natively (page-by-page)
    results = pipe.predict(str(pdf_path))

    # Save each page as Markdown into outdir (this also writes the page assets)
    page_md_paths = []
    for page in results:
        # Each call will produce a 'page_XXXX.md' and a matching 'page_XXXX_files/' folder
        md_path = page.save_to_markdown(save_path=str(outdir))
        page_md_paths.append(Path(md_path))

    # Sort pages by file name (page_0001.md, page_0002.md, …)
    page_md_paths = sorted(page_md_paths, key=lambda p: p.name)

    # Combine into a single document.md
    combined_path = outdir / "document.md"
    with combined_path.open("w", encoding="utf-8") as fout:
        fout.write(f"<!-- Combined by pdf_to_markdown_ppstructv3.py on {datetime.now().isoformat()} -->\n")
        fout.write(f"# {pdf_path.stem}\n\n")

        for idx, md in enumerate(page_md_paths, start=1):
            # Add a clear page separator & header
            fout.write("\n---\n\n")
            fout.write(f"## Page {idx}\n\n")
            with md.open("r", encoding="utf-8") as fin:
                fout.write(fin.read().strip() + "\n")

    print(f"\nDone.\n- Output folder: {outdir}\n- Combined Markdown: {combined_path}")
    print("Note: per-page assets folders (e.g., page_0001_files/) remain alongside 'document.md', "
          "so image links in the combined file just work.")

def main():
    ap = argparse.ArgumentParser(description="Convert a PDF to a single Markdown file using PaddleOCR PP-StructureV3.")
    ap.add_argument("pdf", type=Path, help="Path to input PDF")
    ap.add_argument("--outdir", type=Path, default=None, help="Output folder (default: <PDF_stem>_md)")
    ap.add_argument("--charts", action="store_true", help="Enable chart recognition (disabled by default for stability)")
    args = ap.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: '{pdf_path}' is not a valid PDF path.", file=sys.stderr)
        sys.exit(1)

    outdir = args.outdir or pdf_path.with_suffix("").parent / f"{pdf_path.stem}_md"

    try:
        run_ppstruct(pdf_path, outdir, use_charts=args.charts)
    except Exception as e:
        print("\nPipeline failed.", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        # If you hit the PP-Chart2Table bug, re-run with --charts omitted (default False).
        print("Hint: If this crashed inside chart recognition, try without '--charts'.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
