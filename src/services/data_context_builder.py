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
        "You are **FlavorFlow Craft Assistant** — a senior data analyst embedded "
        "inside a restaurant analytics platform built for the Danish food-service market.\n\n"
        "## Your Domain Knowledge\n"
        "You are an expert on the Loving Loyalty POS ecosystem used by restaurants "
        "in Denmark (currency DKK — Danish Krone, VAT typically 25%). "
        "The platform covers ~1,800+ restaurant locations (mostly in Denmark), "
        "~400K orders, ~2M order line-items, ~88K menu items, ~22K users, "
        "and ~640 promotional campaigns.\n\n"
        "Key concepts you understand deeply:\n"
        "- **BCG Matrix for menus**: Stars (high profit + popular), Plowhorses "
        "(popular but low margin), Puzzles (profitable but underordered), "
        "Dogs (low profit + unpopular).\n"
        "- **Inventory optimization**: EOQ, safety stock, reorder points, "
        "lead time, service level targets, stockout risk.\n"
        "- **Demand forecasting**: Gradient Boosting model trained on order "
        "history with features like day-of-week, hour, price, and item popularity.\n"
        "- **Order types**: Takeaway, Eat-in, Delivery — each with different "
        "behavior patterns.\n"
        "- **Payment methods**: Counter (in-store POS), App (mobile), Card, "
        "Online, etc.\n\n"
        "## Behavioral Rules\n"
        "1. Answer ONLY based on the data context provided below. Never invent data.\n"
        "2. If the data doesn't cover a question, say so and suggest what data would help.\n"
        "3. Keep answers concise but insightful. Use bullet points, tables, and numbers.\n"
        "4. Round numbers sensibly (whole DKK for prices, 1 decimal for percentages).\n"
        "5. When comparing items or restaurants, provide concrete rankings with names.\n"
        "6. Proactively highlight interesting patterns or anomalies you notice.\n"
        "7. Suggest follow-up questions the user might find useful.\n"
        "8. For menu optimization questions, reference BCG categories specifically.\n"
        "9. For inventory questions, mention safety stock, reorder points, and lead times.\n"
        "10. You can do math: calculate margins, growth rates, averages from the raw numbers.\n"
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
        recs_present = (
            (isinstance(recs, pd.DataFrame) and not recs.empty)
            or (isinstance(recs, list) and len(recs) > 0)
        )
        if recs_present:
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
        Build a **deep** statistical summary of every dataset so the LLM
        can answer arbitrary questions about menu items, restaurants,
        orders, campaigns, users, add-ons, etc. without live DB access.

        This replaces the old "table name + row count" stub with real
        pre-computed insights that are compact enough to fit in the
        system prompt (~4-8 k tokens) yet rich enough that the LLM
        can answer most analytical questions.
        """
        self._raw_data["datasets"] = datasets

        lines: List[str] = ["## Deep Data Summary"]

        # ── 1. fct_orders — order-level facts ───────────────────────────
        orders = datasets.get("fct_orders", pd.DataFrame())
        if not orders.empty:
            lines.append(f"\n### Orders ({len(orders):,} total)")

            # date range
            for tc in ("order_time", "created_at", "created"):
                if tc in orders.columns:
                    ts = pd.to_datetime(orders[tc], unit="s", errors="coerce").dropna()
                    if not ts.empty:
                        lines.append(f"- Date range: {ts.min():%Y-%m-%d} → {ts.max():%Y-%m-%d} ({(ts.max()-ts.min()).days:,} days)")
                        # orders per month
                        monthly = ts.dt.to_period("M").value_counts().sort_index()
                        if len(monthly) > 0:
                            lines.append(f"- Monthly order range: {monthly.min():,}–{monthly.max():,} orders/month")
                    break

            # totals / averages
            if "total_amount" in orders.columns:
                amt = pd.to_numeric(orders["total_amount"], errors="coerce").dropna()
                lines.append(f"- Revenue: total={amt.sum():,.0f} DKK, avg={amt.mean():.1f}, median={amt.median():.1f}, std={amt.std():.1f}")
                # percentiles
                lines.append(f"- Revenue percentiles: p25={amt.quantile(.25):.0f}, p75={amt.quantile(.75):.0f}, p95={amt.quantile(.95):.0f} DKK")

            # order type breakdown
            if "type" in orders.columns:
                type_counts = orders["type"].value_counts()
                lines.append("- Order types: " + ", ".join(f"{t}={c:,}" for t, c in type_counts.head(6).items()))

            # payment method
            if "payment_method" in orders.columns:
                pm = orders["payment_method"].value_counts()
                lines.append("- Payment methods: " + ", ".join(f"{m}={c:,}" for m, c in pm.head(6).items()))

            # source / channel
            if "source" in orders.columns:
                src = orders["source"].value_counts()
                lines.append("- Sources: " + ", ".join(f"{s}={c:,}" for s, c in src.head(6).items()))

            if "status" in orders.columns:
                st = orders["status"].value_counts()
                lines.append("- Statuses: " + ", ".join(f"{s}={c:,}" for s, c in st.head(8).items()))

            # top places by order count
            if "place_id" in orders.columns:
                top_places = orders["place_id"].value_counts().head(10)
                lines.append(f"- Top 10 restaurants by orders: {', '.join(f'place_{pid}={cnt:,}' for pid, cnt in top_places.items())}")

        # ── 2. fct_order_items — line-item details ──────────────────────
        oi = datasets.get("fct_order_items", pd.DataFrame())
        if not oi.empty:
            lines.append(f"\n### Order Items ({len(oi):,} line items)")

            if "quantity" in oi.columns:
                qty = pd.to_numeric(oi["quantity"], errors="coerce").dropna()
                lines.append(f"- Quantity stats: total={qty.sum():,.0f}, avg={qty.mean():.2f}, max={qty.max():.0f}")

            if "price" in oi.columns:
                pr = pd.to_numeric(oi["price"], errors="coerce").dropna()
                lines.append(f"- Line-item price: avg={pr.mean():.1f} DKK, median={pr.median():.1f}, max={pr.max():.0f}")

            if "discount_amount" in oi.columns:
                da = pd.to_numeric(oi["discount_amount"], errors="coerce").dropna()
                discounted = (da > 0).sum()
                lines.append(f"- Discounted items: {discounted:,} ({discounted/len(oi)*100:.1f}%), total discount={da.sum():,.0f} DKK")

            if "commission_amount" in oi.columns:
                ca = pd.to_numeric(oi["commission_amount"], errors="coerce").dropna()
                lines.append(f"- Total commission: {ca.sum():,.0f} DKK")

            if "cost" in oi.columns:
                cost = pd.to_numeric(oi["cost"], errors="coerce").dropna()
                non_zero = cost[cost > 0]
                if not non_zero.empty:
                    lines.append(f"- COGS (where reported): avg={non_zero.mean():.1f} DKK, items with cost data={len(non_zero):,}")

        # ── 3. most_ordered — pre-aggregated popularity ─────────────────
        mo = datasets.get("most_ordered", pd.DataFrame())
        if not mo.empty:
            lines.append(f"\n### Most Ordered Items ({len(mo):,} item-restaurant combos)")

            if "order_count" in mo.columns and "item_name" in mo.columns:
                # global top items (sum across restaurants)
                top_global = mo.groupby("item_name")["order_count"].sum().sort_values(ascending=False)
                lines.append("- **Top 25 items by total orders across all restaurants:**")
                for name, cnt in top_global.head(25).items():
                    lines.append(f"  • {name}: {cnt:,} orders")

                # bottom 10
                lines.append("- **Bottom 10 items (least ordered):**")
                for name, cnt in top_global.tail(10).items():
                    lines.append(f"  • {name}: {cnt:,} orders")

            if "place_id" in mo.columns and "order_count" in mo.columns:
                # top restaurants by total item orders
                place_totals = mo.groupby("place_id")["order_count"].sum().sort_values(ascending=False)
                lines.append(f"- Restaurants with highest item orders: {', '.join(f'place_{p}={c:,}' for p, c in place_totals.head(5).items())}")

        # ── 4. dim_items — menu item catalog ────────────────────────────
        items = datasets.get("dim_items", pd.DataFrame())
        if not items.empty:
            lines.append(f"\n### Menu Item Catalog ({len(items):,} items)")

            if "price" in items.columns:
                pr = pd.to_numeric(items["price"], errors="coerce").dropna()
                lines.append(f"- Price range: {pr.min():.0f}–{pr.max():.0f} DKK, avg={pr.mean():.1f}, median={pr.median():.1f}")

            if "status" in items.columns:
                st = items["status"].value_counts()
                lines.append("- Status: " + ", ".join(f"{s}={c:,}" for s, c in st.items()))

            if "type" in items.columns:
                tp = items["type"].value_counts()
                lines.append("- Types: " + ", ".join(f"{t}={c:,}" for t, c in tp.head(6).items()))

            if "vat" in items.columns:
                vat = pd.to_numeric(items["vat"], errors="coerce").dropna()
                lines.append(f"- VAT rates: {', '.join(f'{v}%' for v in sorted(vat.unique())[:6])}")

            # top 15 most expensive items
            if "price" in items.columns and "title" in items.columns:
                expensive = items.dropna(subset=["price"]).nlargest(15, "price")
                lines.append("- **Most expensive items:**")
                for _, r in expensive.iterrows():
                    lines.append(f"  • {r['title']}: {r['price']:.0f} DKK")

                # cheapest active items
                active_items = items[items.get("status", pd.Series()) == "Active"] if "status" in items.columns else items
                cheap = active_items.dropna(subset=["price"])
                cheap = cheap[cheap["price"] > 0].nsmallest(10, "price")
                lines.append("- **Cheapest active items:**")
                for _, r in cheap.iterrows():
                    lines.append(f"  • {r['title']}: {r['price']:.0f} DKK")

        # ── 5. dim_places — restaurant catalog ──────────────────────────
        places = datasets.get("dim_places", pd.DataFrame())
        if not places.empty:
            lines.append(f"\n### Restaurants ({len(places):,} total)")

            if "active" in places.columns:
                active = places[places["active"] == 1] if places["active"].dtype != object else places[places["active"] == "1"]
                lines.append(f"- Active: {len(active):,}, Inactive: {len(places) - len(active):,}")

            if "title" in places.columns:
                lines.append("- **All restaurant names (sample):** " + ", ".join(places["title"].dropna().head(30).tolist()))

            if "country" in places.columns:
                cc = places["country"].value_counts()
                lines.append("- Countries: " + ", ".join(f"{c}={n:,}" for c, n in cc.items()))

            if "currency" in places.columns:
                cur = places["currency"].value_counts()
                lines.append("- Currencies: " + ", ".join(f"{c}={n:,}" for c, n in cur.items()))

            if "street_address" in places.columns:
                addrs = places["street_address"].dropna()
                if not addrs.empty:
                    # extract city from address (last word or area field)
                    if "area" in places.columns:
                        areas = places["area"].dropna().value_counts()
                        lines.append("- Areas: " + ", ".join(f"{a}={n:,}" for a, n in areas.head(15).items()))

            if "cuisine_ids" in places.columns:
                # Just note it exists
                has_cuisine = places["cuisine_ids"].dropna()
                lines.append(f"- {len(has_cuisine)} restaurants have cuisine tags")

            if "rating" in places.columns:
                rat = pd.to_numeric(places["rating"], errors="coerce").dropna()
                if not rat.empty and rat.max() > 0:
                    lines.append(f"- Rating: avg={rat.mean():.1f}, max={rat.max():.1f}")

        # ── 6. dim_campaigns / fct_campaigns — promotions ───────────────
        campaigns = datasets.get("dim_campaigns", datasets.get("fct_campaigns", pd.DataFrame()))
        if not campaigns.empty:
            lines.append(f"\n### Campaigns & Promotions ({len(campaigns):,})")

            if "status" in campaigns.columns:
                cs = campaigns["status"].value_counts()
                lines.append("- Status: " + ", ".join(f"{s}={c:,}" for s, c in cs.items()))

            if "type" in campaigns.columns:
                ct = campaigns["type"].value_counts()
                lines.append("- Types: " + ", ".join(f"{t}={c:,}" for t, c in ct.head(8).items()))

            if "title" in campaigns.columns:
                lines.append("- Recent campaigns: " + ", ".join(f'"{t}"' for t in campaigns["title"].dropna().head(10).tolist()))

            if "discount" in campaigns.columns:
                disc = pd.to_numeric(campaigns["discount"], errors="coerce").dropna()
                if not disc.empty:
                    lines.append(f"- Discount range: {disc.min():.0f}%–{disc.max():.0f}%, avg={disc.mean():.1f}%")

        # ── 7. dim_add_ons — add-on catalog ────────────────────────────
        addons = datasets.get("dim_add_ons", pd.DataFrame())
        if not addons.empty:
            lines.append(f"\n### Add-Ons ({len(addons):,})")
            if "price" in addons.columns:
                ap = pd.to_numeric(addons["price"], errors="coerce").dropna()
                lines.append(f"- Price: avg={ap.mean():.1f} DKK, max={ap.max():.0f}")
                free = (ap == 0).sum()
                lines.append(f"- Free add-ons: {free:,} ({free/len(addons)*100:.0f}%)")
            if "title" in addons.columns and "status" in addons.columns:
                active_ao = addons[addons["status"] == "Active"] if "status" in addons.columns else addons
                lines.append("- Popular add-ons: " + ", ".join(active_ao["title"].dropna().head(15).tolist()))

        # ── 8. dim_users — customer demographics ────────────────────────
        users = datasets.get("dim_users", pd.DataFrame())
        if not users.empty:
            lines.append(f"\n### Users ({len(users):,})")
            if "orders" in users.columns:
                uo = pd.to_numeric(users["orders"], errors="coerce").dropna()
                lines.append(f"- Orders per user: avg={uo.mean():.1f}, median={uo.median():.0f}, max={uo.max():.0f}")
                active_users = (uo > 0).sum()
                lines.append(f"- Users who ordered: {active_users:,} ({active_users/len(users)*100:.1f}%)")
            if "cltv" in users.columns:
                cltv = pd.to_numeric(users["cltv"], errors="coerce").dropna()
                non_zero = cltv[cltv > 0]
                if not non_zero.empty:
                    lines.append(f"- CLTV (lifetime value): avg={non_zero.mean():.0f}, median={non_zero.median():.0f}, max={non_zero.max():.0f} DKK")
            if "source" in users.columns:
                us = users["source"].value_counts()
                lines.append("- Acquisition sources: " + ", ".join(f"{s}={c:,}" for s, c in us.head(5).items()))
            if "country" in users.columns:
                uc = users["country"].value_counts()
                lines.append("- Countries: " + ", ".join(f"{c}={n:,}" for c, n in uc.head(5).items()))

        # ── 9. fct_invoice_items — billing ──────────────────────────────
        invoices = datasets.get("fct_invoice_items", pd.DataFrame())
        if not invoices.empty:
            lines.append(f"\n### Invoice Items ({len(invoices):,})")
            if "amount" in invoices.columns:
                ia = pd.to_numeric(invoices["amount"], errors="coerce").dropna()
                lines.append(f"- Total billed: {ia.sum():,.0f} DKK, avg={ia.mean():.0f}")
            if "description" in invoices.columns:
                desc = invoices["description"].value_counts()
                lines.append("- Product types: " + ", ".join(f"{d}={c}" for d, c in desc.head(8).items()))

        # ── 10. fct_cash_balances — cash management ─────────────────────
        cash = datasets.get("fct_cash_balances", pd.DataFrame())
        if not cash.empty:
            lines.append(f"\n### Cash Balances ({len(cash):,} sessions)")
            if "opening_balance" in cash.columns:
                ob = pd.to_numeric(cash["opening_balance"], errors="coerce").dropna()
                lines.append(f"- Opening balance: avg={ob.mean():.0f} DKK")
            if "closing_balance" in cash.columns:
                cb = pd.to_numeric(cash["closing_balance"], errors="coerce").dropna()
                lines.append(f"- Closing balance: avg={cb.mean():.0f} DKK")

        # ── 11. Cross-table: revenue per restaurant ─────────────────────
        if not orders.empty and "place_id" in orders.columns and "total_amount" in orders.columns:
            orders_c = orders.copy()
            orders_c["total_amount"] = pd.to_numeric(orders_c["total_amount"], errors="coerce")
            rev_by_place = orders_c.groupby("place_id").agg(
                order_count=("total_amount", "count"),
                total_revenue=("total_amount", "sum"),
                avg_order=("total_amount", "mean"),
            ).sort_values("total_revenue", ascending=False)

            # enrich with place names
            if not places.empty and "id" in places.columns and "title" in places.columns:
                pmap = dict(zip(places["id"], places["title"]))
                rev_by_place["name"] = rev_by_place.index.map(lambda x: pmap.get(x, f"place_{x}"))
            else:
                rev_by_place["name"] = rev_by_place.index.map(lambda x: f"place_{x}")

            lines.append(f"\n### Revenue per Restaurant (top 20)")
            for _, r in rev_by_place.head(20).iterrows():
                lines.append(
                    f"  • {r['name']}: {r['total_revenue']:,.0f} DKK revenue, "
                    f"{int(r['order_count']):,} orders, avg {r['avg_order']:.0f} DKK/order"
                )

        # ── 12. Cross-table: item revenue (top 30) ─────────────────────
        if not oi.empty and "item_id" in oi.columns and "price" in oi.columns and "quantity" in oi.columns:
            oi_c = oi.copy()
            oi_c["price"] = pd.to_numeric(oi_c["price"], errors="coerce")
            oi_c["quantity"] = pd.to_numeric(oi_c["quantity"], errors="coerce")
            oi_c["line_total"] = oi_c["price"] * oi_c["quantity"]
            item_rev = oi_c.groupby("item_id").agg(
                total_qty=("quantity", "sum"),
                total_revenue=("line_total", "sum"),
                avg_price=("price", "mean"),
                order_count=("order_id", "nunique") if "order_id" in oi_c.columns else ("item_id", "count"),
            ).sort_values("total_revenue", ascending=False)

            if not items.empty and "id" in items.columns and "title" in items.columns:
                imap = dict(zip(items["id"], items["title"]))
                item_rev["name"] = item_rev.index.map(lambda x: imap.get(x, f"item_{x}"))
            else:
                item_rev["name"] = item_rev.index.map(lambda x: f"item_{x}")

            lines.append(f"\n### Top 30 Items by Revenue")
            for _, r in item_rev.head(30).iterrows():
                lines.append(
                    f"  • {r['name']}: {r['total_revenue']:,.0f} DKK, "
                    f"{int(r['total_qty']):,} units, avg price {r['avg_price']:.0f} DKK"
                )

        # ── 13. Taxonomy / tags ─────────────────────────────────────────
        tax = datasets.get("dim_taxonomy_terms", pd.DataFrame())
        if not tax.empty and "vocabulary" in tax.columns:
            lines.append(f"\n### Taxonomy ({len(tax):,} terms)")
            vocab = tax["vocabulary"].value_counts()
            lines.append("- Vocabularies: " + ", ".join(f"{v}={c}" for v, c in vocab.items()))

        # ── Table schemas (compact) ─────────────────────────────────────
        lines.append("\n### Table Schemas")
        for name, df in datasets.items():
            lines.append(f"- **{name}**: {len(df):,} rows — cols: {', '.join(df.columns[:20])}")

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
            "## How to Use This Data\n"
            "- Cross-reference tables: e.g. link item_id from order_items to dim_items for names/prices.\n"
            "- The 'most_ordered' table is pre-aggregated per restaurant+item — great for popularity questions.\n"
            "- Revenue = price × quantity from order_items (not from orders.total_amount which includes delivery/service charges).\n"
            "- Use the Revenue per Restaurant section for comparative analysis.\n"
            "- BCG classifications come from the ML model trained on this data.\n"
            "- When asked about specific items, search the Top Items lists and item catalog data above.\n"
            "- If a question requires data you don't have, say so and suggest what to look at.\n"
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
