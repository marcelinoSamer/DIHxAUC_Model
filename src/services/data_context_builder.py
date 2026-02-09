"""
File: data_context_builder.py
Description: Builds rich data context from analysis results for the LLM.
Dependencies: pandas
Author: FlavorFlow Team

Converts raw DataFrames and analysis dictionaries into a compact
natural-language summary that fits inside the LLM system prompt.

This is the bridge between "what the data says" and "what the LLM knows".
Every piece of item-level, pricing, and performance data that the analysis
services produce is surfaced here so the LLM can answer specific questions.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _fmt(val: Any, decimals: int = 1) -> str:
    """Format a number to a string, rounding nicely."""
    v = _safe_float(val)
    if v == int(v) and decimals <= 1:
        return f"{int(v):,}"
    return f"{v:,.{decimals}f}"


def _df_get(row: Any, col: str, fallback: str = "?") -> Any:
    """Safely get a value from a DataFrame row or dict."""
    if isinstance(row, dict):
        return row.get(col, fallback)
    try:
        v = row[col] if col in row.index else fallback
        if pd.isna(v):
            return fallback
        return v
    except Exception:
        return fallback


class DataContextBuilder:
    """
    Builds a token-budget-aware context string from analysis results.

    The system prompt tells the LLM *what it is* and *what it knows*.
    The data context fills in the "what it knows" part with live numbers
    — including individual item names, prices, revenues, and rankings.
    """

    # ── class-level defaults ─────────────────────────────────────────────

    MAX_TOP_ITEMS = 25          # How many "top X" items to include
    MAX_ALERTS = 20             # Max inventory alerts to embed
    MAX_BCG_ITEMS = 20          # Max items per BCG category to list
    MAX_PRICING = 25            # Max pricing suggestions to include

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
        "- When asked about 'performance', give a holistic overview covering "
        "revenue, top items, BCG health, inventory status, and demand trends.\n"
    )

    def __init__(self) -> None:
        self._sections: Dict[str, str] = {}
        self._raw_data: Dict[str, Any] = {}

    # ── ingestion methods ────────────────────────────────────────────────

    def ingest_inventory_results(self, results: Dict[str, Any]) -> None:
        """
        Ingest results from InventoryAnalysisService.run_full_analysis().

        Extracts: behavior, demand model metrics, inventory alerts
        (critical + low + excess with item names), optimisation params.
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
            n = min(len(top_items), self.MAX_TOP_ITEMS)
            lines.append(f"\n### Top {n} Items by Order Volume")
            for item in top_items[:n]:
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

        # ── inventory alerts (with FULL item-level detail) ──
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

            # critical items
            if not critical.empty:
                n = min(len(critical), self.MAX_ALERTS)
                lines.append(f"\n**Critical items (top {n}):**")
                for _, row in critical.head(n).iterrows():
                    lines.append(
                        f"  • {_df_get(row, 'title', _df_get(row, 'item_id'))}: "
                        f"current={_df_get(row, 'current_stock')}, "
                        f"reorder_point={_df_get(row, 'reorder_point')}"
                    )

            # low stock items
            if not low.empty:
                n = min(len(low), 15)
                lines.append(f"\n**Low-stock items (top {n}):**")
                for _, row in low.head(n).iterrows():
                    lines.append(
                        f"  • {_df_get(row, 'title', _df_get(row, 'item_id'))}: "
                        f"current={_df_get(row, 'current_stock')}, "
                        f"reorder_point={_df_get(row, 'reorder_point')}"
                    )

            # excess stock items
            if not excess.empty:
                n = min(len(excess), 10)
                lines.append(f"\n**Excess-stock items (top {n}):**")
                for _, row in excess.head(n).iterrows():
                    lines.append(
                        f"  • {_df_get(row, 'title', _df_get(row, 'item_id'))}: "
                        f"current={_df_get(row, 'current_stock')}, "
                        f"reorder_point={_df_get(row, 'reorder_point')}"
                    )

        # ── full inventory analysis table (per-item stock levels) ──
        analysis_df = inv.get("analysis", pd.DataFrame())
        if isinstance(analysis_df, pd.DataFrame) and not analysis_df.empty:
            lines.append(f"\n### Inventory Optimization Detail ({len(analysis_df)} items)")
            for _, row in analysis_df.head(30).iterrows():
                title = _df_get(row, "title", _df_get(row, "item_id"))
                stock = _df_get(row, "optimal_stock_level", _df_get(row, "reorder_point"))
                safety = _df_get(row, "safety_stock", "?")
                lines.append(f"  • {title}: optimal_stock={stock}, safety_stock={safety}")

        # ── inventory optimization params ──
        opt_summary = inv.get("summary", {})
        params = opt_summary.get("parameters", {})
        if params:
            lines.append("\n### Inventory Optimization Parameters")
            lines.append(f"- Lead time: {params.get('lead_time_days', '?')} days")
            try:
                sl = float(params.get('service_level', 0))
                lines.append(f"- Service level target: {sl * 100:.0f}%")
            except (ValueError, TypeError):
                lines.append(f"- Service level target: {params.get('service_level', '?')}")
            lines.append(
                f"- Total items analyzed: "
                f"{opt_summary.get('total_items_analyzed', '?')}"
            )

        self._sections["inventory"] = "\n".join(lines)

    def ingest_bcg_results(self, results: Dict[str, Any]) -> None:
        """
        Ingest results from MenuAnalysisService.

        Extracts: BCG breakdown WITH individual items per category,
        top stars/dogs/plowhorses/puzzles by revenue, pricing suggestions
        with DKK amounts, cluster info, and recommendations.
        """
        self._raw_data["bcg"] = results
        lines: List[str] = ["## Menu Engineering (BCG Matrix)"]

        # ── executive summary ──
        summary = results.get("executive_summary", results)
        bcg_breakdown = summary.get("bcg_breakdown", {})
        data_overview = summary.get("data_overview", {})

        if data_overview:
            total_items = data_overview.get("total_items", "?")
            total_restaurants = data_overview.get("total_restaurants", "?")
            total_orders = data_overview.get("total_orders", "?")
            total_campaigns = data_overview.get("total_campaigns", "?")
            lines.append("\n### Network Overview")
            lines.append(f"- Total menu items analyzed: {total_items}")
            lines.append(f"- Total restaurants: {total_restaurants}")
            if total_orders != "?":
                lines.append(f"- Total orders in dataset: {_fmt(total_orders, 0)}")
            if total_campaigns and total_campaigns != "?":
                lines.append(f"- Active campaigns: {total_campaigns}")

        if bcg_breakdown:
            lines.append("\n### BCG Category Counts")
            lines.append(f"- ⭐ Stars (high popularity + high price): {bcg_breakdown.get('stars', '?')}")
            lines.append(f"- 🐴 Plowhorses (high popularity + low price): {bcg_breakdown.get('plowhorses', '?')}")
            lines.append(f"- ❓ Puzzles (low popularity + high price): {bcg_breakdown.get('puzzles', '?')}")
            lines.append(f"- 🐕 Dogs (low popularity + low price): {bcg_breakdown.get('dogs', '?')}")

        # ── BCG per-item breakdown (THE BIG ADDITION) ──
        bcg_df = results.get("bcg_classification", pd.DataFrame())
        if isinstance(bcg_df, pd.DataFrame) and not bcg_df.empty:
            # Compute network-wide totals from the BCG data
            total_rev = _safe_float(bcg_df.get("total_revenue", pd.Series([0])).sum())
            total_ord = _safe_float(bcg_df.get("total_orders", pd.Series([0])).sum())
            if total_rev > 0:
                lines.append(f"\n### Network Financial Summary")
                lines.append(f"- Total revenue (all items): {_fmt(total_rev, 0)} DKK")
                lines.append(f"- Total orders (all items): {_fmt(total_ord, 0)}")
                avg_rev = total_rev / max(len(bcg_df), 1)
                lines.append(f"- Average revenue per item: {_fmt(avg_rev, 1)} DKK")

            # Sort once by revenue for reuse
            rev_col = "total_revenue" if "total_revenue" in bcg_df.columns else None
            cat_col = "category" if "category" in bcg_df.columns else "bcg_category"
            if rev_col and cat_col in bcg_df.columns:
                bcg_sorted = bcg_df.sort_values(rev_col, ascending=False)

                for emoji, label, cat_name in [
                    ("⭐", "Stars", "Star"),
                    ("🐴", "Plowhorses", "Plowhorse"),
                    ("❓", "Puzzles", "Puzzle"),
                    ("🐕", "Dogs", "Dog"),
                ]:
                    subset = bcg_sorted[
                        bcg_sorted[cat_col].str.contains(cat_name, case=False, na=False)
                    ]
                    if subset.empty:
                        continue
                    n = min(len(subset), self.MAX_BCG_ITEMS)
                    cat_rev = _safe_float(subset[rev_col].sum())
                    cat_ord = _safe_float(subset.get("total_orders", pd.Series([0])).sum())
                    lines.append(
                        f"\n### {emoji} Top {n} {label} "
                        f"(total: {len(subset)} items, "
                        f"rev: {_fmt(cat_rev, 0)} DKK, "
                        f"orders: {_fmt(cat_ord, 0)})"
                    )
                    for _, row in subset.head(n).iterrows():
                        title = _df_get(row, "title", _df_get(row, "item_name", "?"))
                        price = _fmt(_df_get(row, "avg_price", _df_get(row, "price", 0)))
                        rev = _fmt(_df_get(row, "total_revenue", 0), 0)
                        orders = _fmt(_df_get(row, "total_orders", 0), 0)
                        lines.append(
                            f"  • {title} — price: {price} DKK, "
                            f"revenue: {rev} DKK, orders: {orders}"
                        )

        # ── BCG metrics / thresholds ──
        bcg_metrics = results.get("bcg_metrics", {})
        if bcg_metrics:
            pop_t = bcg_metrics.get("popularity_threshold")
            price_t = bcg_metrics.get("price_threshold")
            if pop_t is not None or price_t is not None:
                lines.append("\n### BCG Classification Thresholds")
                if pop_t is not None:
                    lines.append(f"- Popularity threshold: {_fmt(pop_t)} orders")
                if price_t is not None:
                    lines.append(f"- Price threshold: {_fmt(price_t)} DKK")

        # ── pricing suggestions (item-level DKK amounts) ──
        pricing = results.get("pricing_suggestions", pd.DataFrame())
        if isinstance(pricing, pd.DataFrame) and not pricing.empty:
            n = min(len(pricing), self.MAX_PRICING)
            total_gain = _safe_float(pricing.get("revenue_gain", pd.Series([0])).sum())
            lines.append(
                f"\n### Pricing Suggestions "
                f"({len(pricing)} items, potential gain: {_fmt(total_gain, 0)} DKK)"
            )
            # Sort by revenue_gain descending
            if "revenue_gain" in pricing.columns:
                pricing_sorted = pricing.sort_values("revenue_gain", ascending=False)
            else:
                pricing_sorted = pricing
            for _, row in pricing_sorted.head(n).iterrows():
                title = _df_get(row, "title", _df_get(row, "item_name", "?"))
                cur = _fmt(_df_get(row, "current_price", _df_get(row, "avg_price", 0)))
                sug = _fmt(_df_get(row, "suggested_price", 0))
                gain = _fmt(_df_get(row, "revenue_gain", 0), 0)
                cat = _df_get(row, "category", "?")
                lines.append(
                    f"  • {title} ({cat}): current {cur} DKK → suggested {sug} DKK "
                    f"(+{gain} DKK potential)"
                )
        elif isinstance(pricing, list) and len(pricing) > 0:
            lines.append(f"\n### Pricing Suggestions ({len(pricing)} items)")
            for p in pricing[:self.MAX_PRICING]:
                title = p.get("title", p.get("item_name", "?"))
                cur = _fmt(p.get("current_price", p.get("avg_price", 0)))
                sug = _fmt(p.get("suggested_price", 0))
                gain = _fmt(p.get("revenue_gain", 0), 0)
                lines.append(
                    f"  • {title}: current {cur} DKK → suggested {sug} DKK "
                    f"(+{gain} DKK potential)"
                )

        # ── pricing opportunity from executive summary ──
        pricing_opp = summary.get("pricing_opportunity", {})
        if pricing_opp:
            lines.append("\n### Pricing Opportunity (aggregate)")
            lines.append(
                f"- Total potential revenue gain: "
                f"{_fmt(pricing_opp.get('total_revenue_gain', 0), 0)} DKK"
            )
            lines.append(
                f"- Items that should be repriced: "
                f"{pricing_opp.get('items_to_reprice', '?')}"
            )

        # ── cluster info ──
        clusters = results.get("clustered_items", pd.DataFrame())
        cluster_profiles = results.get("cluster_profiles", {})
        if isinstance(clusters, pd.DataFrame) and not clusters.empty and "cluster" in clusters.columns:
            n_clusters = clusters["cluster"].nunique()
            lines.append(f"\n### Item Clusters ({n_clusters} groups)")
            for cid in sorted(clusters["cluster"].unique()):
                grp = clusters[clusters["cluster"] == cid]
                grp_rev = _safe_float(grp.get("total_revenue", pd.Series([0])).sum())
                avg_price = _safe_float(grp.get("avg_price", pd.Series([0])).mean())
                lines.append(
                    f"- Cluster {cid}: {len(grp)} items, "
                    f"avg price {_fmt(avg_price)} DKK, "
                    f"total revenue {_fmt(grp_rev, 0)} DKK"
                )
                # Show top 5 items from each cluster
                if "total_revenue" in grp.columns:
                    top5 = grp.nlargest(5, "total_revenue")
                else:
                    top5 = grp.head(5)
                for _, row in top5.iterrows():
                    title = _df_get(row, "title", _df_get(row, "item_name", "?"))
                    lines.append(f"    · {title}")

        # ── demand prediction model (menu-side) ──
        prediction = results.get("prediction", {})
        if prediction:
            pred_metrics = prediction.get("metrics", {})
            if pred_metrics:
                lines.append("\n### Menu Demand Prediction Model")
                lines.append(f"- MAE: {pred_metrics.get('mae', '?')}")
                lines.append(f"- RMSE: {pred_metrics.get('rmse', '?')}")
                lines.append(f"- R²: {pred_metrics.get('r2', '?')}")
                lines.append(
                    f"- Training samples: {pred_metrics.get('training_samples', '?')}"
                )
            feat = prediction.get("feature_importance", [])
            if feat:
                lines.append("  Top features:")
                for fi in feat[:5]:
                    lines.append(
                        f"    · {fi.get('feature', '?')}: "
                        f"{_safe_float(fi.get('importance', 0)):.3f}"
                    )

        # ── strategic recommendations ──
        recs = results.get("recommendations", [])
        recs_present = (
            (isinstance(recs, pd.DataFrame) and not recs.empty)
            or (isinstance(recs, list) and len(recs) > 0)
        )
        if recs_present:
            lines.append("\n### Strategic Recommendations")
            if isinstance(recs, pd.DataFrame):
                for _, row in recs.head(12).iterrows():
                    lines.append(
                        f"- [{_df_get(row, 'priority')}] "
                        f"{_df_get(row, 'category')}: "
                        f"{_df_get(row, 'recommendation')}"
                    )
            elif isinstance(recs, list):
                for rec in recs[:12]:
                    lines.append(
                        f"- [{rec.get('priority', '?')}] "
                        f"{rec.get('category', '?')}: "
                        f"{rec.get('recommendation', '?')}"
                    )

        self._sections["bcg"] = "\n".join(lines)

    def ingest_raw_data_summary(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """
        Ingest a summary of the raw datasets including key statistics
        (not just shapes — also revenue totals and top items where possible).
        """
        self._raw_data["dataset_summary"] = {
            name: {"rows": len(df), "columns": list(df.columns)}
            for name, df in datasets.items()
        }

        lines: List[str] = ["## Available Datasets"]
        for name, info in self._raw_data["dataset_summary"].items():
            lines.append(f"- **{name}**: {info['rows']:,} rows, {len(info['columns'])} cols")

        # ── extract key stats from raw dataframes ──
        orders_df = datasets.get("orders") or datasets.get("fct_orders")
        if orders_df is not None and isinstance(orders_df, pd.DataFrame):
            lines.append(f"\n### Orders Dataset Highlights")
            lines.append(f"- Total orders: {len(orders_df):,}")
            if "total_price" in orders_df.columns:
                total = _safe_float(orders_df["total_price"].sum())
                avg = _safe_float(orders_df["total_price"].mean())
                lines.append(f"- Total revenue: {_fmt(total, 0)} DKK")
                lines.append(f"- Average order value: {_fmt(avg, 1)} DKK")
            if "place_id" in orders_df.columns:
                lines.append(
                    f"- Unique restaurants with orders: "
                    f"{orders_df['place_id'].nunique()}"
                )

        items_df = datasets.get("items") or datasets.get("dim_items")
        if items_df is not None and isinstance(items_df, pd.DataFrame):
            lines.append(f"\n### Items Dataset Highlights")
            lines.append(f"- Total items: {len(items_df):,}")
            if "price" in items_df.columns:
                avg_price = _safe_float(items_df["price"].mean())
                max_price = _safe_float(items_df["price"].max())
                lines.append(f"- Average item price: {_fmt(avg_price, 1)} DKK")
                lines.append(f"- Max item price: {_fmt(max_price, 1)} DKK")

        places_df = datasets.get("places") or datasets.get("dim_places")
        if places_df is not None and isinstance(places_df, pd.DataFrame):
            lines.append(f"\n### Places (Restaurants) Highlights")
            lines.append(f"- Total restaurants: {len(places_df):,}")

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

        prompt = "\n".join(parts)
        return prompt

    def build_compact_context(self, max_chars: int = 12000) -> str:
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
