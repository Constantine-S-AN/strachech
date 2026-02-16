"""Capture report screenshots with Playwright when optional dependency is available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture summary/window/execution-quality screenshots from site/report/index.html."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root directory. Default: current directory.",
    )
    parser.add_argument(
        "--report",
        default="site/report/index.html",
        help="Report page path or URL. Default: site/report/index.html",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/screens"),
        help="Output screenshot directory. Default: docs/assets/screens",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=1600,
        help="Viewport width in pixels. Default: 1600",
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=1000,
        help="Viewport height in pixels. Default: 1000",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=1200,
        help="Wait time after navigation in milliseconds. Default: 1200",
    )
    return parser.parse_args()


def resolve_report_target(project_root: Path, report_arg: str) -> tuple[str | None, Path | None]:
    if report_arg.startswith("http://") or report_arg.startswith("https://"):
        return report_arg, None
    if report_arg.startswith("file://"):
        return report_arg, None

    candidate_path = Path(report_arg)
    if not candidate_path.is_absolute():
        candidate_path = project_root / candidate_path
    candidate_path = candidate_path.resolve()
    if not candidate_path.exists():
        return None, candidate_path
    return candidate_path.as_uri(), candidate_path


def import_playwright() -> tuple[Any, Any] | tuple[None, None]:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None, None
    return sync_playwright, PlaywrightError


def section_locator(page: Any, heading_text: str) -> Any | None:
    heading = page.locator("h2", has_text=heading_text).first
    if heading.count() == 0:
        return None

    parent_section = heading.locator("xpath=ancestor::section[1]")
    if parent_section.count() > 0:
        return parent_section.first
    return heading


def capture_sections(
    *,
    page: Any,
    output_dir: Path,
) -> None:
    targets = [
        ("summary", "Summary"),
        ("window_table", "Window Metrics"),
        ("execution_quality", "Execution Quality"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for filename_stem, heading_text in targets:
        destination_path = output_dir / f"{filename_stem}.png"
        locator = section_locator(page, heading_text)
        if locator is None:
            print(
                f"[screens] section not found for heading '{heading_text}'. "
                f"Capturing full-page fallback: {destination_path}"
            )
            page.screenshot(path=str(destination_path), full_page=True)
            continue

        locator.scroll_into_view_if_needed(timeout=4000)
        page.wait_for_timeout(250)
        locator.screenshot(path=str(destination_path))
        print(f"[screens] captured {heading_text}: {destination_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args()
    project_root = args.root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    )

    target_url, target_path = resolve_report_target(
        project_root=project_root, report_arg=str(args.report)
    )
    if target_url is None:
        print("[screens] report page not found. Skipping screenshot capture.")
        print(f"[screens] Missing path: {target_path}")
        print("[screens] Run: python scripts/build_site.py --root .")
        return 0

    sync_playwright_fn, playwright_error_type = import_playwright()
    if sync_playwright_fn is None:
        print("[screens] Playwright Python package is not installed. Skipping screenshot capture.")
        print("[screens] Install with: pip install playwright")
        print("[screens] Then install browser: python -m playwright install chromium")
        return 0

    try:
        with sync_playwright_fn() as playwright_manager:
            browser = playwright_manager.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={
                    "width": int(args.viewport_width),
                    "height": int(args.viewport_height),
                }
            )
            page.goto(target_url, wait_until="networkidle")
            page.wait_for_timeout(int(args.wait_ms))
            capture_sections(page=page, output_dir=output_dir)
            browser.close()
    except playwright_error_type as error:  # type: ignore[arg-type]
        print(
            "[screens] Playwright browser dependency is unavailable. Skipping screenshot capture."
        )
        print(f"[screens] Detail: {error}")
        print("[screens] Install browser with: python -m playwright install chromium")
        return 0
    except Exception as error:  # noqa: BLE001
        print("[screens] Unexpected screenshot error. Skipping to avoid CI failure.")
        print(f"[screens] Detail: {error}")
        return 0

    print(f"[screens] screenshots saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
