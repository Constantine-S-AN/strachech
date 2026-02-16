"""Build a GitHub Pages static demo site under site/."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from stratcheck.core.experiments import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static /site pages for GitHub Pages.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Reports output directory. Default: reports.",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path("site"),
        help="Site output directory. Default: site.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=240,
        help="Number of bars for demo data generation. Default: 240.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for demo data generation. Default: 7.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path
    return project_root / raw_path


def run_command(command: list[str], project_root: Path) -> None:
    subprocess.run(command, cwd=project_root, check=True)


def get_git_commit_hash(project_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def ensure_results_jsonl(project_root: Path, reports_dir: Path) -> None:
    results_path = reports_dir / "results.jsonl"
    if results_path.exists():
        return

    configs_dir = project_root / "configs" / "examples"
    if not configs_dir.exists():
        return

    runner = ExperimentRunner(configs_dir=configs_dir, output_dir=reports_dir)
    runner.run_all()


def run_release_demo_commands(
    *,
    project_root: Path,
    reports_dir: Path,
    periods: int,
    seed: int,
) -> None:
    demo_data_path = project_root / "data" / "QQQ.csv"
    original_data: bytes | None = None
    had_original_data = demo_data_path.exists()
    if had_original_data:
        original_data = demo_data_path.read_bytes()

    try:
        run_command(
            [
                sys.executable,
                "scripts/make_demo_assets.py",
                "--output",
                str(demo_data_path),
                "--periods",
                str(periods),
                "--seed",
                str(seed),
            ],
            project_root,
        )
        run_command(
            [
                sys.executable,
                "-m",
                "stratcheck",
                "demo",
                "--output",
                str(reports_dir / "release_demo.html"),
                "--periods",
                str(periods),
                "--seed",
                str(seed),
            ],
            project_root,
        )
        ensure_results_jsonl(project_root=project_root, reports_dir=reports_dir)
        run_command(
            [
                sys.executable,
                "-m",
                "stratcheck",
                "dashboard",
                "--results-jsonl",
                str(reports_dir / "results.jsonl"),
                "--db",
                str(reports_dir / "paper_trading.sqlite"),
                "--output",
                str(reports_dir / "dashboard.html"),
                "--reports-dir",
                str(reports_dir),
            ],
            project_root,
        )
    finally:
        if had_original_data and original_data is not None:
            demo_data_path.write_bytes(original_data)
        elif not had_original_data and demo_data_path.exists():
            demo_data_path.unlink()


def collect_report_files_from_results(reports_dir: Path) -> list[Path]:
    report_paths: list[Path] = []
    results_path = reports_dir / "results.jsonl"
    if not results_path.exists():
        return report_paths

    with results_path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            report_name = str(payload.get("report_path", "")).strip()
            if not report_name:
                continue
            candidate = reports_dir / report_name
            if candidate.exists() and candidate not in report_paths:
                report_paths.append(candidate)

    return report_paths


def copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    if not source_dir.exists():
        return

    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


def build_site_assets(
    *,
    project_root: Path,
    reports_dir: Path,
    site_dir: Path,
) -> tuple[Path, Path]:
    if site_dir.exists():
        shutil.rmtree(site_dir)

    (site_dir / "assets").mkdir(parents=True, exist_ok=True)
    (site_dir / "report").mkdir(parents=True, exist_ok=True)
    (site_dir / "dashboard").mkdir(parents=True, exist_ok=True)
    (site_dir / "case-study").mkdir(parents=True, exist_ok=True)

    reports_assets = reports_dir / "assets"
    copy_tree_contents(source_dir=reports_assets, target_dir=site_dir / "assets")

    docs_images = project_root / "docs" / "images"
    if docs_images.exists():
        copy_tree_contents(source_dir=docs_images, target_dir=site_dir / "assets" / "docs")
    docs_assets = project_root / "docs" / "assets"
    if docs_assets.exists():
        copy_tree_contents(source_dir=docs_assets, target_dir=site_dir / "assets")

    release_report_source = reports_dir / "release_demo.html"
    if not release_report_source.exists():
        msg = f"release report not found: {release_report_source}"
        raise FileNotFoundError(msg)

    dashboard_source = reports_dir / "dashboard.html"
    if not dashboard_source.exists():
        msg = f"dashboard report not found: {dashboard_source}"
        raise FileNotFoundError(msg)

    release_report_target = site_dir / "release_demo.html"
    dashboard_target = site_dir / "dashboard.html"

    shutil.copy2(release_report_source, release_report_target)
    shutil.copy2(dashboard_source, dashboard_target)

    for report_path in collect_report_files_from_results(reports_dir=reports_dir):
        shutil.copy2(report_path, site_dir / report_path.name)

    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    return release_report_target, dashboard_target


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_index_html(
    *,
    commit_hash: str,
    generated_time: str,
    periods: int,
    seed: int,
) -> str:
    command_lines = [
        (
            "python scripts/make_demo_assets.py "
            f"--output data/QQQ.csv --periods {periods} --seed {seed}"
        ),
        (
            "python -m stratcheck demo "
            f"--output reports/release_demo.html --periods {periods} --seed {seed}"
        ),
        (
            "python -m stratcheck dashboard "
            "--results-jsonl reports/results.jsonl "
            "--db reports/paper_trading.sqlite "
            "--output reports/dashboard.html --reports-dir reports"
        ),
        "python scripts/build_site.py",
    ]
    reproducible_commands = "\n".join(command_lines)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Stratcheck Demo Landing</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --accent: #0f766e;
      --surface: #ffffff;
      --background: #f8fafc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f0fdfa, var(--background));
    }}
    main {{ width: min(960px, 92vw); margin: 28px auto; }}
    .block {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 14px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 4.4vw, 42px); }}
    h2 {{ margin: 0 0 8px; font-size: 22px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.55; }}
    ul {{ margin: 0; padding-left: 18px; color: var(--muted); }}
    .cards {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      background: #f8fafc;
    }}
    .button {{
      display: inline-block;
      text-decoration: none;
      color: #ffffff;
      background: var(--accent);
      border-radius: 10px;
      padding: 9px 12px;
      margin-top: 10px;
      font-weight: 700;
      font-size: 14px;
    }}
    code {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      margin: 10px 0 0;
      font-size: 13px;
    }}
    .meta {{ font-size: 13px; color: #64748b; }}
  </style>
</head>
<body>
  <main>
    <section class=\"block\">
      <h1>Stratcheck Live Demo</h1>
      <p>Static deliverable package for report + dashboard, built from existing CLI workflows.</p>
    </section>

    <section class=\"block\">
      <h2>Open Deliverables</h2>
      <div class=\"cards\">
        <article class=\"card\">
          <h3>Release Report</h3>
          <p>
            Generated by
            <code>python -m stratcheck demo --output reports/release_demo.html</code>
          </p>
          <a class=\"button\" href=\"report/\">Open Report</a>
        </article>
        <article class=\"card\">
          <h3>Dashboard</h3>
          <p>
            Generated by
            <code>python -m stratcheck dashboard ... --output reports/dashboard.html</code>
          </p>
          <a class=\"button\" href=\"dashboard/\">Open Dashboard</a>
        </article>
        <article class=\"card\">
          <h3>Case Study</h3>
          <p>
            Interview-focused story: Problem, Solution, Results, and minimal reproduction flow.
          </p>
          <a class=\"button\" href=\"case-study/\">Open Case Study</a>
        </article>
      </div>
    </section>

    <section class=\"block\">
      <h2>Three Highlights</h2>
      <ul>
        <li>Single-command reproducibility from data generation to shareable HTML outputs.</li>
        <li>
          Report and dashboard are packaged as static assets with no runtime server dependency.
        </li>
        <li>
          CI-compatible structure for GitHub Pages deployment from the
          <code>site/</code> directory.
        </li>
      </ul>
    </section>

    <section class=\"block\">
      <p class=\"meta\">Git commit: <code>{commit_hash}</code></p>
      <p class=\"meta\">Generated at: <code>{generated_time}</code></p>
      <details>
        <summary>Reproduction Commands</summary>
        <pre>{reproducible_commands}</pre>
      </details>
    </section>
  </main>
</body>
</html>
"""


def render_redirect_html(*, title: str, target_href: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"refresh\" content=\"0; url={target_href}\" />
  <title>{title}</title>
</head>
<body>
  <p>Redirecting to <a href=\"{target_href}\">{target_href}</a> ...</p>
</body>
</html>
"""


_CASE_STUDY_IMAGE_PATTERN = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")


def _rewrite_case_study_image_source(raw_source: str) -> str:
    source = raw_source.strip()
    if source.startswith("./assets/"):
        return "../assets/" + source.removeprefix("./assets/")
    if source.startswith("assets/"):
        return "../assets/" + source.removeprefix("assets/")
    if source.startswith("./images/"):
        return "../assets/docs/" + source.removeprefix("./images/")
    if source.startswith("images/"):
        return "../assets/docs/" + source.removeprefix("images/")
    if source.startswith("../docs/images/"):
        return "../assets/docs/" + source.removeprefix("../docs/images/")
    if source.startswith("/docs/images/"):
        return "../assets/docs/" + source.removeprefix("/docs/images/")
    return source


def _render_inline_markdown(text: str) -> str:
    segments = re.split(r"(`[^`]+`)", text)
    rendered: list[str] = []
    for segment in segments:
        if not segment:
            continue
        if segment.startswith("`") and segment.endswith("`") and len(segment) >= 2:
            rendered.append(f"<code>{html.escape(segment[1:-1])}</code>")
            continue
        rendered.append(html.escape(segment))
    return "".join(rendered)


def _markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []
    code_lines: list[str] = []
    in_code_block = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph_text = " ".join(part.strip() for part in paragraph_lines if part.strip())
        if paragraph_text:
            output.append(f"<p>{_render_inline_markdown(paragraph_text)}</p>")
        paragraph_lines.clear()

    def flush_list() -> None:
        if not list_items:
            return
        output.append("<ul>")
        for item in list_items:
            output.append(f"<li>{_render_inline_markdown(item)}</li>")
        output.append("</ul>")
        list_items.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if in_code_block:
            if stripped.startswith("```"):
                output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code_block = False
            else:
                code_lines.append(line)
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            in_code_block = True
            continue

        image_match = _CASE_STUDY_IMAGE_PATTERN.match(stripped)
        if image_match:
            flush_paragraph()
            flush_list()
            alt_text = image_match.group(1).strip()
            source = _rewrite_case_study_image_source(image_match.group(2))
            caption = html.escape(alt_text)
            output.append('<figure class="shot">')
            output.append(
                f'<img src="{html.escape(source)}" alt="{caption}" '
                'loading="lazy" decoding="async" />'
            )
            if caption:
                output.append(f"<figcaption>{caption}</figcaption>")
            output.append("</figure>")
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            flush_list()
            output.append(f"<h1>{_render_inline_markdown(stripped[2:].strip())}</h1>")
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            output.append(f"<h2>{_render_inline_markdown(stripped[3:].strip())}</h2>")
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flush_list()
            output.append(f"<h3>{_render_inline_markdown(stripped[4:].strip())}</h3>")
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            list_items.append(stripped[2:].strip())
            continue

        flush_list()
        paragraph_lines.append(stripped)

    if in_code_block:
        output.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
    flush_paragraph()
    flush_list()
    return "\n".join(output)


def render_case_study_html(*, content_html: str, generated_time: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Stratcheck Case Study</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --line: #cbd5e1;
      --surface: #ffffff;
      --background: #f8fafc;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f0fdfa, var(--background));
    }}
    main {{
      width: min(980px, 92vw);
      margin: 24px auto 40px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 22px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 14px;
    }}
    .home-link {{
      text-decoration: none;
      color: #ffffff;
      background: var(--accent);
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 14px;
      font-weight: 700;
    }}
    h1 {{ margin: 6px 0 10px; font-size: 34px; }}
    h2 {{ margin: 20px 0 10px; font-size: 24px; }}
    h3 {{ margin: 18px 0 8px; font-size: 20px; }}
    p, li {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 16px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      display: grid;
      gap: 8px;
    }}
    code {{
      background: #0f172a;
      color: #e2e8f0;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 14px;
    }}
    pre {{
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      margin: 8px 0 0;
      font-size: 14px;
    }}
    .shot {{
      margin: 14px 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8fafc;
      padding: 10px;
    }}
    .shot img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      display: block;
    }}
    .shot figcaption {{
      margin-top: 8px;
      color: #64748b;
      font-size: 14px;
    }}
    .meta {{
      color: #64748b;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <div class=\"topbar\">
      <div class=\"meta\">Stratcheck Case Study</div>
      <a class=\"home-link\" href=\"../index.html\">Back to Home</a>
    </div>
    <p class=\"meta\">Generated at: <code>{generated_time}</code></p>
    {content_html}
  </main>
</body>
</html>
"""


def build_case_study_page(*, project_root: Path, site_dir: Path, generated_time: str) -> None:
    source_path = project_root / "docs" / "case-study.md"
    if not source_path.exists():
        msg = f"case study source not found: {source_path}"
        raise FileNotFoundError(msg)
    markdown_text = source_path.read_text(encoding="utf-8")
    content_html = _markdown_to_html(markdown_text=markdown_text)
    page_html = render_case_study_html(content_html=content_html, generated_time=generated_time)
    write_text(site_dir / "case-study" / "index.html", page_html)


def build_site(
    *,
    project_root: Path,
    reports_dir: Path,
    site_dir: Path,
    periods: int,
    seed: int,
) -> Path:
    run_release_demo_commands(
        project_root=project_root,
        reports_dir=reports_dir,
        periods=periods,
        seed=seed,
    )
    build_site_assets(project_root=project_root, reports_dir=reports_dir, site_dir=site_dir)

    generated_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    commit_hash = get_git_commit_hash(project_root=project_root)
    build_case_study_page(
        project_root=project_root, site_dir=site_dir, generated_time=generated_time
    )

    write_text(
        site_dir / "index.html",
        render_index_html(
            commit_hash=commit_hash,
            generated_time=generated_time,
            periods=periods,
            seed=seed,
        ),
    )
    write_text(
        site_dir / "report" / "index.html",
        render_redirect_html(title="Release Report", target_href="../release_demo.html"),
    )
    write_text(
        site_dir / "dashboard" / "index.html",
        render_redirect_html(title="Dashboard", target_href="../dashboard.html"),
    )

    return site_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args()
    project_root = args.root.resolve()
    reports_dir = resolve_path(project_root, args.reports_dir)
    site_dir = resolve_path(project_root, args.site_dir)

    site_path = build_site(
        project_root=project_root,
        reports_dir=reports_dir,
        site_dir=site_dir,
        periods=int(args.periods),
        seed=int(args.seed),
    )
    print(f"Site generated: {site_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
