"""
File: data_context_builder.py
Description: Builds rich data context from analysis results for the LLM.
Dependencies: pandas
Author: FlavorFlow Team

Converts raw DataFrames and analysis dictionaries into a compact
natural-language summary that fits inside the LLM system prompt.

This is the bridge between "what the data says" and "what the LLM knows".
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime


class DataContextBuilder:
    """
    Builds a token-budget-aware context string from analysis results.

    The system prompt tells the LLM *what it is* and *what it knows*.
    The data context fills in the "what it knows" part with live numbers.

    Usage:
        >>> builder = DataContextBuilder()
        >>> builder.ingest_inventory_results(results_dict)
        >>> builder.ingest_bcg_results(bcg_dict)
        >>> system_prompt = builder.build_system_prompt()
    """

    # ── class-level defaults ─────────────────────────────────────────────

    MAX_TOP_ITEMS = 15          # How many "top X" items to include
    MAX_ALERTS = 20             # Max inventory alerts to embed

    _PERSONA = (
        "You are FlavorFlow Craft Assistant — an expert data analyst embedded "
        "inside a restaurant analytics platform. You help restaurant managers "
        "understand their menu performance, customer behavior, inventory levels, "
        "and demand forecasts.\n\n"
        "Rules:\n"
        "- Answer ONLY based on the data context provided below.\n"
        "- If you don't know or the data doesn't cover it, say so honestly.\n"
        "- Keep answers concise but insightful. Use bullet points and numbers.\n"
        "- When citing numbers, round to sensible precision.\n"
        "- Currency is DKK (Danish Krone).\n"
        "- You may suggest follow-up questions the user might want to ask.\n"
    )

    def __init__(self) -> None:
        self._sections: Dict[str, str] = {}
        self._raw_data: Dict[str, Any] = {}

    # ── ingestion methods ────────────────────────────────────────────────

    def ingest_inventory_results(self, results: Dict[str, Any]) -> None:
        """
        Ingest results from InventoryAnalysisService.run_full_analysis().

        Extracts: behavior, demand model metrics, inventory alerts,
        executive summary.
        """
        self._raw_data["inventory"] = results

        lines: List[str] = ["## Inventory & Demand Analysis"]

        # ── meta ──
        meta = results.get("meta", {})
        if meta:
            lines.append(f"Analysis completed: {meta.get('completed_at', 'N/A')}")
            lines.append(
                f"Pipeline runtime: {meta.get('total_time_seconds', 0):.1f}s"
            )

        # ── customer behavior ──
        behavior = results.get("behavior", {})
        summary = behavior.get("summary", {})

        temporal = summary.get("temporal_insights", {})
        if temporal:
            lines.append("\n### Customer Behavior — Temporal")
            lines.append(f"- Peak ordering hour: {temporal.get('peak_hour_label', '?')}")
            lines.append(f"- Peak day: {temporal.get('peak_day', '?')}")
            lines.append(f"- Weekend share: {temporal.get('weekend_pct', '?')}%")
            lines.append(
                f"- Avg daily orders (network): {temporal.get('avg_orders_per_day', '?')}"
            )

        purchase = summary.get("purchase_insights", {})
        if purchase:
            lines.append("\n### Customer Behavior — Purchases")
            lines.append(
                f"- Avg items per order: {purchase.get('avg_items_per_order', '?')}"
            )
            lines.append(
                f"- Avg quantity per order: {purchase.get('avg_quantity_per_order', '?')}"
            )
            lines.append(
                f"- Avg order value: {purchase.get('avg_order_value', '?')} DKK"
            )
            lines.append(
                f"- Median order value: {purchase.get('median_order_value', '?')} DKK"
            )

        # ── top items from purchase patterns ──
        purchase_patterns = behavior.get("purchase", {})
        top_items = purchase_patterns.get("top_items", [])
        if top_items:
            lines.append(
                f"\n### Top {min(len(top_items), self.MAX_TOP_ITEMS)} Items by Order Volume"
            )
            for item in top_items[: self.MAX_TOP_ITEMS]:
                lines.append(
                    f"- {item.get('title', item.get('item_id', '?'))}: "
                    f"{item.get('total_quantity', '?')} units, "
                    f"{item.get('order_count', '?')} orders"
                )

        # ── demand model ──
        demand = results.get("demand_model", {})
        metrics = demand.get("metrics", {})
        if metrics:
            lines.append("\n### Demand Forecasting Model")
            lines.append(f"- Algorithm: Gradient Boosting Regressor")
            lines.append(f"- MAE: {metrics.get('mae', '?')} units")
            lines.append(f"- RMSE: {metrics.get('rmse', '?')} units")
            lines.append(f"- R² score: {metrics.get('r2', '?')}")

        feat_imp = demand.get("feature_importance", [])
        if feat_imp:
            lines.append("\n### Feature Importance (top 5)")
            for f in feat_imp[:5]:
                lines.append(
                    f"- {f.get('feature', '?')}: {f.get('importance', 0):.3f}"
                )

        # ── inventory alerts ──
        inv = results.get("inventory", {})
        alerts_df = inv.get("alerts", pd.DataFrame())
        if isinstance(alerts_df, pd.DataFrame) and not alerts_df.empty:
            critical = alerts_df[alerts_df["status"].str.contains("Critical", na=False)]
            low = alerts_df[alerts_df["status"].str.contains("Low", na=False)]
            excess = alerts_df[alerts_df["status"].str.contains("Excess", na=False)]

            lines.append("\n### Inventory Alerts Summary")
            lines.append(f"- 🔴 Critical (stockout risk): {len(critical)} items")
            lines.append(f"- 🟠 Low stock (reorder soon): {len(low)} items")
            lines.append(f"- 🔵 Excess (overstock risk): {len(excess)} items")

            if not critical.empty:
                lines.append(
                    f"\nTop {min(len(critical), 10)} critical items:"
                )
                for _, row in critical.head(10).iterrows():
                    lines.append(
                        f"  • {row.get('title', row.get('item_id', '?'))}: "
                        f"current={row.get('current_stock', '?')}, "
                        f"needed={row.get('reorder_point', '?')}"
                    )

        # ── inventory optimization params ──
        opt_summary = inv.get("summary", {})
        params = opt_summary.get("parameters", {})
        if params:
            lines.append("\n### Inventory Optimization Parameters")
            lines.append(f"- Lead time: {params.get('lead_time_days', '?')} days")
            lines.append(
                f"- Service level target: "
                f"{params.get('service_level', 0) * 100:.0f}%"
            )
            lines.append(
                f"- Total items analyzed: "
                f"{opt_summary.get('total_items_analyzed', '?')}"
            )

        self._sections["inventory"] = "\n".join(lines)

    def ingest_bcg_results(self, results: Dict[str, Any]) -> None:
        """
        Ingest results from MenuAnalysisService.

        Extracts: BCG breakdown, top stars/dogs, recommendations,
        pricing suggestions.
        """
        self._raw_data["bcg"] = results
        lines: List[str] = ["## Menu Engineering (BCG Matrix)"]

        # summary
        summary = results.get("executive_summary", results)
        bcg_breakdown = summary.get("bcg_breakdown", {})
        if bcg_breakdown:
            lines.append("\n### BCG Breakdown")
            lines.append(f"- ⭐ Stars: {bcg_breakdown.get('stars', '?')}")
            lines.append(
                f"- 🐴 Plowhorses: {bcg_breakdown.get('plowhorses', '?')}"
            )
            lines.append(f"- ❓ Puzzles: {bcg_breakdown.get('puzzles', '?')}")
            lines.append(f"- 🐕 Dogs: {bcg_breakdown.get('dogs', '?')}")

        data_overview = summary.get("data_overview", {})
        if data_overview:
            lines.append("\n### Data Overview")
            lines.append(
                f"- Total menu items: {data_overview.get('total_items', '?')}"
            )
            lines.append(
                f"- Total restaurants: {data_overview.get('total_restaurants', '?')}"
            )

        # top recommendations
        recs = results.get("recommendations", [])
        if recs:
            lines.append("\n### Strategic Recommendations")
            if isinstance(recs, pd.DataFrame):
                for _, row in recs.head(8).iterrows():
                    lines.append(
                        f"- [{row.get('priority', '?')}] "
                        f"{row.get('category', '?')}: "
                        f"{row.get('recommendation', '?')}"
                    )
            elif isinstance(recs, list):
                for rec in recs[:8]:
                    lines.append(
                        f"- [{rec.get('priority', '?')}] "
                        f"{rec.get('category', '?')}: "
                        f"{rec.get('recommendation', '?')}"
                    )

        self._sections["bcg"] = "\n".join(lines)

    def ingest_raw_data_summary(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """
        Ingest a lightweight summary of the raw dataset shapes.

        Useful even when no analysis has been run yet so the LLM
        knows what data is available.
        """
        self._raw_data["dataset_summary"] = {
            name: {"rows": len(df), "columns": list(df.columns)}
            for name, df in datasets.items()
        }

        lines: List[str] = ["## Available Datasets"]
        for name, info in self._raw_data["dataset_summary"].items():
            lines.append(f"- **{name}**: {info['rows']:,} rows, {len(info['columns'])} cols")

        self._sections["datasets"] = "\n".join(lines)

    def ingest_custom_section(self, key: str, markdown: str) -> None:
        """Add an arbitrary markdown section to the context."""
        self._sections[key] = markdown

    # ── prompt building ──────────────────────────────────────────────────

    def build_system_prompt(self) -> str:
        """
        Assemble the full system prompt (persona + data sections).

        Returns:
            A single string ready to be used as the ``system`` message.
        """
        parts: List[str] = [self._PERSONA, "---\n# DATA CONTEXT\n"]

        # Deterministic ordering
        order = ["datasets", "inventory", "bcg"]
        for key in order:
            if key in self._sections:
                parts.append(self._sections[key])
                parts.append("")  # blank line

        # Any extra sections
        for key, text in self._sections.items():
            if key not in order:
                parts.append(text)
                parts.append("")

        parts.append(
            "---\n"
            "Use the data above to answer the user's questions.\n"
            "If a question falls outside this data, say so clearly."
        )
        return "\n".join(parts)

    def build_compact_context(self, max_chars: int = 6000) -> str:
        """
        Build a shorter context for token-constrained models.

        Prioritises inventory alerts and BCG breakdown, then truncates.
        """
        full = self.build_system_prompt()
        if len(full) <= max_chars:
            return full
        return full[:max_chars] + "\n\n[… context truncated for token budget]"

    # ── utilities ────────────────────────────────────────────────────────

    def get_section_keys(self) -> List[str]:
        """Return the list of ingested context sections."""
        return list(self._sections.keys())

    def clear(self) -> None:
        """Reset all ingested context."""
        self._sections.clear()
        self._raw_data.clear()

    def __repr__(self) -> str:
        sections = ", ".join(self._sections.keys()) or "empty"
        return f"DataContextBuilder(sections=[{sections}])"
