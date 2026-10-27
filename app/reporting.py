import io
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from app.config import ROOT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def _render_donut_chart(violation_rate: float) -> io.BytesIO:
    """Render a crisp donut chart for the violation percentage card."""
    fig, ax = plt.subplots(figsize=(1.6, 1.6), subplot_kw=dict(aspect="equal"), dpi=160)
    rate = max(0.0, min(100.0, float(violation_rate)))
    rem = 100.0 - rate if rate < 100.0 else 0.001
    
    # Donut slices: rate (red), remainder (soft gray)
    slice_colors = ["#E53935", "#E2E8F0"] if rate > 0 else ["#22C55E", "#E2E8F0"]
    ax.pie(
        [rate if rate > 0 else 100, rem if rate > 0 else 0],
        colors=slice_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1.5),
    )
    center_text = f"{rate:.1f}%"
    ax.text(
        0, 0, center_text,
        ha="center", va="center",
        fontsize=12, fontweight="bold",
        color="#1E293B",
    )
    plt.tight_layout(pad=0.1)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_pie_chart(violations_by_type: list, total_violations: int) -> io.BytesIO:
    """Render the pie chart of violation categories matching the reference design."""
    fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=160)
    
    active_items = [v for v in violations_by_type if v.get("count", 0) > 0]
    
    if not active_items or total_violations == 0:
        # Zero violations state: 100% compliant green circle
        ax.pie([1], colors=["#10B981"], wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
        ax.text(0, 0, "100%\nCompliant", ha="center", va="center", fontsize=11, fontweight="bold", color="#065F46")
        ax.axis("equal")
    else:
        values = [v["count"] for v in active_items]
        colors_list = [v.get("color", "#EF4444") for v in active_items]
        
        wedges, texts, autotexts = ax.pie(
            values,
            colors=colors_list,
            autopct="%1.1f%%",
            pctdistance=0.68,
            startangle=90,
            counterclock=False,
            textprops=dict(color="white", fontweight="bold", fontsize=9),
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        ax.axis("equal")
    
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _render_hourly_bar_chart(hourly_timeline: list, peak_violations: int) -> io.BytesIO:
    """Render the hourly non-compliance trend bar chart with peak annotation."""
    fig, ax = plt.subplots(figsize=(4.6, 2.7), dpi=160)
    
    # Filter hours: prioritize shift range 08:00 to 18:00 or active hours
    shift_hours = [h for h in hourly_timeline if 8 <= int(h["hour"][:2]) <= 17]
    if not shift_hours:
        shift_hours = hourly_timeline[:12]
        
    slots = []
    for item in shift_hours:
        h = int(item["hour"][:2])
        slots.append(f"{h:02d}:00-\n{(h+1)%24:02d}:00")
        
    counts = [item.get("violations", 0) for item in shift_hours]
    max_c = max(counts) if counts else 0
    
    # Soft blue bars with red highlight for peak period
    bar_colors = ["#EF4444" if (c == max_c and c > 0) else "#60A5FA" for c in counts]
    
    x = range(len(slots))
    bars = ax.bar(x, counts, color=bar_colors, width=0.55, zorder=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(slots, fontsize=6.5, color="#475569")
    ax.set_ylabel("Non-Compliant Detections", fontsize=7.5, fontweight="bold", color="#334155")
    ax.set_xlabel("Time (Hour)", fontsize=8, fontweight="bold", color="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    
    # Value labels on top of each bar
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.25,
                str(c),
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold",
                color="#1E293B",
            )
            
    # Draw "Highest Violation Period" callout above peak bar
    if max_c > 0:
        peak_idx = counts.index(max_c)
        y_top = max_c + 1.2
        ax.annotate(
            "Highest Violation Period\n↓",
            xy=(peak_idx, max_c + 0.3),
            xytext=(peak_idx, y_top),
            ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color="#DC2626",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=0.8),
        )
        ax.set_ylim(0, max_c + 2.8)
    else:
        ax.set_ylim(0, 5)
        
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    
    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_daily_pdf(date_str: str, summary: dict) -> bytes:
    """Generate an executive-grade Daily PPE Report matching the reference visual template."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()

    # Metrics extraction
    total = summary.get("total_inspections", 0)
    compliant = summary.get("compliant_count", 0)
    violations = summary.get("violation_count", 0)
    violation_rate = summary.get("violation_rate", 0.0)
    violations_by_type = summary.get("violations_by_type", [])
    hourly = summary.get("hourly", [])
    peak_v = summary.get("peak_violations", 0)
    recent = summary.get("recent_violations", [])

    # Format Date for Header
    try:
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = dt_obj.strftime("%d %b %Y")
    except Exception:
        formatted_date = date_str

    report_time = datetime.now().strftime("%H:%M")

    # Use rolling window info if available in summary
    window_start = summary.get("window_start", "")
    window_end = summary.get("window_end", "")
    if window_start and window_end:
        period_line = f"<b>Period</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {window_start} &#8594; {window_end}<br/>"
    else:
        period_line = f"<b>Report Time</b> : {report_time}<br/>"

    elements = []

    # 1. TOP NAVY HEADER BANNER
    header_data = [
        [
            Paragraph(
                """<b><font size="19" color="#FFFFFF">DAILY PPE REPORT</font></b><br/>
                <font size="9.5" color="#93C5FD">Automated PPE Violation Detection System</font>""",
                styles["Normal"],
            ),
            Paragraph(
                f"""<font size="8" color="#CBD5E1">
                <b>Date</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: {formatted_date}<br/>
                {period_line}
                <b>Location</b> &nbsp;&nbsp;: Production Hall<br/>
                <b>Camera</b> &nbsp;&nbsp;&nbsp;&nbsp;: Gate-01 (Main)<br/>
                <b>System</b> &nbsp;&nbsp;&nbsp;&nbsp;: PPE Guard
                </font>""",
                styles["Normal"],
            ),
        ]
    ]
    header_table = Table(header_data, colWidths=[330, 217])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0B2545")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    elements.append(header_table)
    elements.append(Spacer(1, 14))

    # SECTION 1: OVERALL SUMMARY
    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0B2545"),
    )
    elements.append(Paragraph("<b>1. OVERALL SUMMARY</b>", section_heading_style))
    elements.append(Spacer(1, 6))

    # Generate Donut Chart Image
    donut_buf = _render_donut_chart(violation_rate)
    donut_img = Image(donut_buf, width=70, height=70)

    # 4 Summary Cards side by side
    card_col_w = 136.75  # 547 / 4
    card_title_style = ParagraphStyle(
        "CardTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        alignment=1,  # Center
        textColor=colors.HexColor("#334155"),
    )

    summary_cards_data = [
        [
            Paragraph("Total Detections", card_title_style),
            Paragraph("Compliant Detections", card_title_style),
            Paragraph("Non-Compliant Detections", card_title_style),
            Paragraph("Violation Rate", card_title_style),
        ],
        [
            Paragraph(f"""<font size="24" color="#0B2545"><b>{total}</b></font>""", card_title_style),
            Paragraph(f"""<font size="24" color="#16A34A"><b>{compliant}</b></font>""", card_title_style),
            Paragraph(f"""<font size="24" color="#DC2626"><b>{violations}</b></font>""", card_title_style),
            donut_img,
        ],
    ]
    summary_table = Table(summary_cards_data, colWidths=[card_col_w] * 4)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F0FDF4")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#FEF2F2")),
                ("BACKGROUND", (3, 0), (3, -1), colors.HexColor("#FFFBEB")),
                ("BOX", (0, 0), (0, -1), 1, colors.HexColor("#E2E8F0")),
                ("BOX", (1, 0), (1, -1), 1, colors.HexColor("#BBF7D0")),
                ("BOX", (2, 0), (2, -1), 1, colors.HexColor("#FECACA")),
                ("BOX", (3, 0), (3, -1), 1, colors.HexColor("#FDE68A")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.append(summary_table)
    elements.append(Spacer(1, 14))

    # SECTIONS 2 & 3: CHARTS SIDE BY SIDE
    pie_buf = _render_pie_chart(violations_by_type, violations)
    pie_img = Image(pie_buf, width=225, height=170)

    bar_buf = _render_hourly_bar_chart(hourly, peak_v)
    bar_img = Image(bar_buf, width=295, height=170)

    # Build Legend Table for Pie Chart
    legend_rows = []
    for item in violations_by_type:
        bullet_hex = item.get("color", "#EF4444")
        lbl = item.get("label", "")
        cnt = item.get("count", 0)
        pct = item.get("percentage", 0.0)
        legend_rows.append(
            [
                Paragraph(f"""<font color="{bullet_hex}">●</font> <b>{lbl}</b>""", styles["Normal"]),
                Paragraph(f"<b>{cnt} ({pct:.1f}%)</b>", styles["Normal"]),
            ]
        )
    if not legend_rows:
        legend_rows.append([Paragraph("No violations recorded", styles["Normal"]), Paragraph("0 (0%)", styles["Normal"])])

    legend_table = Table(legend_rows, colWidths=[150, 80])
    legend_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )

    charts_content = [
        [
            Paragraph("<b>2. VIOLATIONS BY TYPE (NON-COMPLIANCE)</b>", section_heading_style),
            Paragraph("<b>3. HOURLY NON-COMPLIANCE TREND</b>", section_heading_style),
        ],
        [
            pie_img,
            bar_img,
        ],
        [
            legend_table,
            Paragraph("", styles["Normal"]),
        ],
    ]
    charts_table = Table(charts_content, colWidths=[245, 302])
    charts_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FFFFFF")),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#FFFFFF")),
                ("BOX", (0, 0), (0, -1), 1, colors.HexColor("#E2E8F0")),
                ("BOX", (1, 0), (1, -1), 1, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elements.append(charts_table)
    elements.append(Spacer(1, 14))

        # Add Captured Violation Photographic Evidences
    violation_dir = ROOT / "config" / "violations"
    evidence_items = []
    if recent:
        for r in recent:
            fname = r.get("image_filename")
            if fname:
                img_path = violation_dir / fname
                if img_path.is_file():
                    evidence_items.append((img_path, r))

    # SECTION 4: RECENT INCIDENT LOG TABLE
    elements.append(Paragraph("<b>4. RECENT VIOLATION INCIDENTS LOG</b>", section_heading_style))
    elements.append(Spacer(1, 6))

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#FFFFFF"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1E293B"),
    )

    table_data = [
        [
            Paragraph("Incident ID", table_header_style),
            Paragraph("Time", table_header_style),
            Paragraph("Camera View", table_header_style),
            Paragraph("Flagged Violations", table_header_style),
            Paragraph("Confidence", table_header_style),
            Paragraph("Status", table_header_style),
        ]
    ]

    if not recent:
        table_data.append(
            [
                Paragraph("-", table_cell_style),
                Paragraph("-", table_cell_style),
                Paragraph("-", table_cell_style),
                Paragraph("Zero violations recorded on this date.", table_cell_style),
                Paragraph("-", table_cell_style),
                Paragraph("100% Compliant", table_cell_style),
            ]
        )
    else:
        # If no snapshots, show 5 incidents so the whole report fits cleanly on 1 page!
        # If snapshots exist, show 10 incidents on page 1, followed by page break for snapshots.
        row_limit = 10 if evidence_items else 5
        for r in recent[:row_limit]:
            viol_lbls = ", ".join(r.get("violations", [])) or "non-compliant"
            conf_str = "-"
            dets = r.get("details", [])
            if dets and isinstance(dets, list):
                c = dets[0].get("confidence")
                if c is not None:
                    conf_str = f"{c*100:.1f}%"
            worker_id = r.get("worker_id")
            w_label = f"#{r.get('id', '')}" + (f" (W#{worker_id})" if worker_id is not None else "")
            table_data.append(
                [
                    Paragraph(w_label, table_cell_style),
                    Paragraph(str(r.get("time", "")), table_cell_style),
                    Paragraph(str(r.get("view", "")).capitalize(), table_cell_style),
                    Paragraph(viol_lbls, table_cell_style),
                    Paragraph(conf_str, table_cell_style),
                    Paragraph("<font color='#DC2626'><b>Flagged</b></font>", table_cell_style),
                ]
            )

    incidents_table = Table(table_data, colWidths=[65, 55, 75, 202, 75, 75])
    incidents_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(KeepTogether(incidents_table))

    # SECTION 5: CAPTURED VIOLATION VISUAL EVIDENCES
    if evidence_items:
        # Start snapshots on clean dedicated page
        elements.append(PageBreak())
        elements.append(Paragraph("<b>5. CAPTURED SAFETY VIOLATION VISUAL EVIDENCES</b>", section_heading_style))
        elements.append(Spacer(1, 6))

        evidence_grid = []
        row_cells = []
        for img_path, r in evidence_items[:16]:
            try:
                img_widget = Image(str(img_path), width=260, height=146)
                w_id = r.get("worker_id")
                w_tag = f"Worker #{w_id}" if w_id is not None else "Worker"
                caption = Paragraph(
                    f"<b>{w_tag}</b> &bull; {r.get('time', '')} &bull; {', '.join(r.get('violations', []))}",
                    table_cell_style,
                )
                cell_box = [img_widget, Spacer(1, 2), caption]
                row_cells.append(cell_box)
                if len(row_cells) == 2:
                    evidence_grid.append(row_cells)
                    row_cells = []
            except Exception as exc:
                print(f"[Reporting] Error embedding snapshot image: {exc}")

        if row_cells:
            row_cells.append([Paragraph("", table_cell_style)])
            evidence_grid.append(row_cells)

        if evidence_grid:
            evidence_table = Table(evidence_grid, colWidths=[273, 273])
            evidence_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            elements.append(evidence_table)
    else:
        no_snap_style = ParagraphStyle(
            "NoSnap",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#64748B"),
        )
        sec5_box = [
            Spacer(1, 10),
            Paragraph("<b>5. CAPTURED SAFETY VIOLATION VISUAL EVIDENCES</b>", section_heading_style),
            Spacer(1, 4),
            Paragraph(
                "&#8212; No violation snapshots captured for this reporting period. "
                "Snapshots are automatically saved when a worker is detected without required PPE "
                "while cameras are active. &#8212;",
                no_snap_style,
            ),
        ]
        elements.append(KeepTogether(sec5_box))

    # Build Document
    doc.build(elements)
    return buffer.getvalue()
