"""Report generation package."""

from stratcheck.report.builder import ReportBuilder
from stratcheck.report.generator import render_html_report
from stratcheck.report.plots import generate_performance_plots

__all__ = ["ReportBuilder", "generate_performance_plots", "render_html_report"]
