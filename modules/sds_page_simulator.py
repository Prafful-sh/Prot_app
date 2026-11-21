"""
SDS-PAGE Gel Simulation Module (Modern Lab GelDoc Edition)
==========================================================
Simulates high-fidelity, laboratory-grade destained Tris-Glycine SDS-PAGE gels:
- Realistic polyacrylamide translucent matrix with destained Coomassie R-250 (~50–100 ng/band sensitivity)
  or ultra-sensitive Silver Stain (~0.2–1.0 ng/band sensitivity, 50–100× more sensitive)
- Diffused Gaussian-like protein band dispersion (core + soft feathering)
- Official Bio-Rad Precision Plus Protein™ Dual Color / Kaleidoscope 10-band MW ladder in the LAST well
  (250, 150, 100, 75 [pink ref], 50 [intense blue ref], 37, 25 [pink ref], 20, 15, 10 kDa)
- Authentic E. coli BL21(DE3) host cell proteome background & Ni-NTA chaperone/His-rich trace contaminants
  (RpoB/C, AceE, ArnA, DnaK, GroEL, Tig, EF-Tu, OmpF/OmpA, GapA, SlyD; Bolanos-Garcia & Davies 2006)
- Ferguson-consistent sigmoidal molecular sieve migration (Rf) across 4%–25% polyacrylamide gels
"""

import math
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

# Official Bio-Rad Precision Plus Protein™ Recombinant Standards (10, 15, 20, 25, 37, 50, 75, 100, 150, 250 kDa)
# Dual Color (#1610374): 8 blue bands + 2 pink high-intensity reference bands at 25 kDa & 75 kDa (50 kDa is intense blue ref)
# Kaleidoscope (#1610375): 250 blue, 150 purple, 100 blue, 75 pink ref, 50 blue ref, 37 green, 25 pink ref, 20 blue, 15 blue, 10 yellow
STANDARD_LADDERS = {
    "precision_plus": {
        "name": "Bio-Rad Precision Plus Protein Dual Color",
        "catalog": "Bio-Rad #1610374",
        "bands": [
            {"mw_kda": 250.0, "label": "250 kDa", "prominent": False, "color_type": "blue"},
            {"mw_kda": 150.0, "label": "150 kDa", "prominent": False, "color_type": "blue"},
            {"mw_kda": 100.0, "label": "100 kDa", "prominent": False, "color_type": "blue"},
            {"mw_kda": 75.0,  "label": "75 kDa",  "prominent": True,  "color_type": "pink_ref"},  # High-intensity pink reference band
            {"mw_kda": 50.0,  "label": "50 kDa",  "prominent": True,  "color_type": "blue"},      # High-intensity blue reference band
            {"mw_kda": 37.0,  "label": "37 kDa",  "prominent": False, "color_type": "blue"},
            {"mw_kda": 25.0,  "label": "25 kDa",  "prominent": True,  "color_type": "pink_ref"},  # High-intensity pink reference band
            {"mw_kda": 20.0,  "label": "20 kDa",  "prominent": False, "color_type": "blue"},
            {"mw_kda": 15.0,  "label": "15 kDa",  "prominent": False, "color_type": "blue"},
            {"mw_kda": 10.0,  "label": "10 kDa",  "prominent": False, "color_type": "blue"}
        ]
    },
    "kaleidoscope": {
        "name": "Bio-Rad Precision Plus Protein Kaleidoscope",
        "catalog": "Bio-Rad #1610375",
        "bands": [
            {"mw_kda": 250.0, "label": "250 kDa", "prominent": False, "color_type": "blue"},
            {"mw_kda": 150.0, "label": "150 kDa", "prominent": False, "color_type": "purple_ref"},
            {"mw_kda": 100.0, "label": "100 kDa", "prominent": False, "color_type": "blue"},
            {"mw_kda": 75.0,  "label": "75 kDa",  "prominent": True,  "color_type": "pink_ref"},
            {"mw_kda": 50.0,  "label": "50 kDa",  "prominent": True,  "color_type": "blue"},
            {"mw_kda": 37.0,  "label": "37 kDa",  "prominent": False, "color_type": "green_ref"},
            {"mw_kda": 25.0,  "label": "25 kDa",  "prominent": True,  "color_type": "pink_ref"},
            {"mw_kda": 20.0,  "label": "20 kDa",  "prominent": False, "color_type": "blue"},
            {"mw_kda": 15.0,  "label": "15 kDa",  "prominent": False, "color_type": "blue"},
            {"mw_kda": 10.0,  "label": "10 kDa",  "prominent": False, "color_type": "yellow_ref"}
        ]
    }
}


def calculate_sds_rf(mw_kda: float, gel_pct: float) -> float:
    """
    Calculates relative gel migration coordinate (Rf in [0.06, 0.94]) on Tris-Glycine SDS-PAGE
    between the stacking/resolving interface (y = 0.075) and Bromophenol Blue dye front (y = 0.940).
    Follows Ferguson retardation (log Rf = log Y0 - Kr * %T) and Ogston–Rodbard–Chrambach
    sigmoidal molecular sieving:
        rf_raw = 1 / (1 + (MW / MW50(%T))^gamma)
    Verified against Bio-Rad Precision Plus migration charts across 4%–22% acrylamide:
      - 6% gel:  250 kDa enters well (~0.32); <15 kDa proteins compress near the dye front (~0.87–0.90)
      - 12% gel: 37–50 kDa central markers sit right near the middle of the resolving gel (~0.46–0.54)
      - 15% gel: 250 kDa stays near the top (~0.13); 14.3 kDa Lysozyme (~0.69) sits cleanly right below
                 the 15 kDa marker (~0.68) and well above the 10 kDa marker (~0.76)
    Strictly monotonic decreasing in both mw_kda and gel_pct.
    """
    if mw_kda <= 0:
        return 0.935

    safe_gel_pct = max(3.0, min(30.0, float(gel_pct)))
    mw_50 = 47.0 * ((11.2 / safe_gel_pct) ** 1.45)
    gamma = 1.24
    rf_raw = 1.0 / (1.0 + (mw_kda / mw_50) ** gamma)

    # Map [0, 1] sieve mobility into resolving gel slab [0.080, 0.935]
    rf = 0.080 + (rf_raw * 0.850)
    return round(float(np.clip(rf, 0.060, 0.940)), 4)


def _compute_non_overlapping_label_positions(rfs: List[float], min_gap: float = 0.022) -> List[float]:
    """
    Nudges right-margin text annotation y-coordinates slightly when extreme gel percentages
    (e.g., 4% or 22%) compress top or bottom ladder bands, keeping band shapes at exact Rf
    while preventing label text overlap.
    """
    if not rfs:
        return []
    pos = list(rfs)
    # Forward pass (top -> bottom)
    pos[0] = max(0.085, pos[0])
    for idx in range(1, len(pos)):
        if pos[idx] - pos[idx - 1] < min_gap:
            pos[idx] = pos[idx - 1] + min_gap
    # Backward pass if bottom label pushed past dye front
    if pos[-1] > 0.930:
        pos[-1] = 0.930
        for idx in range(len(pos) - 2, -1, -1):
            if pos[idx + 1] - pos[idx] < min_gap:
                pos[idx] = pos[idx + 1] - min_gap
    return pos


def generate_sds_page_figure(
    target_mw_kda: float,
    gel_pct: float = 12.0,
    include_dimer: bool = False,
    stain_type: str = "Coomassie Blue",
    show_purification_stages: bool = True
) -> go.Figure:
    """
    Constructs a modern, publication-ready visual simulation of a destained SDS-PAGE gel.
    Places the Bio-Rad Precision Plus Dual Color ladder in the LAST well, and target protein
    fractions in the preceding lanes.
    - Coomassie Brilliant Blue R-250 detection limit: ~50–100 ng protein/band.
    - Silver Stain detection limit: ~0.2–1.0 ng protein/band (~50–100x higher sensitivity),
      revealing low-abundance E. coli host cell background bands and trace co-eluting chaperones.
    - Under standard reducing SDS-PAGE (+DTT / beta-mercaptoethanol, include_dimer=False),
      non-covalent oligomers dissociate into monomers. Setting include_dimer=True optionally
      displays a minor unreduced disulfide / SDS-resistant dimer band at 2 * target_mw_kda.
    """
    fig = go.Figure()

    # Staining color presets
    is_silver = "Silver" in stain_type

    # Background styling: clean destained polyacrylamide cassette
    # Coomassie R-250: clear crisp destained slab (#F8FAFC)
    # Silver Stain: authentic subtle warm amber/sepia-white slab (#FFFDF6)
    gel_slab_bg = "#FFFDF6" if is_silver else "#F8FAFC"
    gel_border = "#D1D5DB" if is_silver else "#CBD5E1"

    ladder_bands = STANDARD_LADDERS["precision_plus"]["bands"]

    # Define Lanes (Ladder always in LAST well)
    if show_purification_stages:
        lanes = [
            {"id": 1, "name": "Lysate (Input)", "sub": "Total Extract", "type": "sample"},
            {"id": 2, "name": "Soluble", "sub": "Clarified", "type": "sample"},
            {"id": 3, "name": "Flowthru (FT)", "sub": "Unbound", "type": "sample"},
            {"id": 4, "name": "Elution", "sub": "Target Peak", "type": "sample"},
            {"id": 5, "name": "Ladder", "sub": "MW Marker", "type": "ladder"}
        ]
    else:
        lanes = [
            {"id": 1, "name": "Purified Target", "sub": "Elution Peak", "type": "sample"},
            {"id": 2, "name": "Ladder", "sub": "MW Marker", "type": "ladder"}
        ]

    num_lanes = len(lanes)
    lane_width = 1.0 / (num_lanes + 0.6)

    # 1. Outer Transilluminator Frame / Border
    fig.add_shape(
        type="rect",
        x0=-0.02, y0=-0.08, x1=1.04, y1=1.03,
        fillcolor=gel_slab_bg,
        line=dict(color=gel_border, width=2),
        layer="below"
    )

    # 2. Stacking Gel (4% T, pH 6.8) / Resolving Gel (%T, pH 8.8) Interface (y = 0.075)
    fig.add_shape(
        type="line",
        x0=0.01, y0=0.075, x1=1.01, y1=0.075,
        line=dict(color="rgba(148, 163, 184, 0.45)", width=1.5, dash="dot"),
        layer="below"
    )
    fig.add_annotation(
        x=0.02, y=0.065,
        text="Stacking (4%, pH 6.8) / Resolving (pH 8.8) Interface",
        showarrow=False,
        font=dict(size=9, color="#57534E" if is_silver else "#52606D", family="IBM Plex Sans, Helvetica Neue, Arial, sans-serif"),
        xanchor="left"
    )

    # 3. Dye Front Line (Bromophenol Blue) at bottom (y = 0.940)
    dye_color = "rgba(30, 41, 59, 0.22)" if is_silver else "rgba(37, 99, 235, 0.28)"
    fig.add_shape(
        type="rect",
        x0=0.02, y0=0.935, x1=1.00, y1=0.945,
        fillcolor=dye_color,
        line_width=0,
        layer="below"
    )
    fig.add_annotation(
        x=0.5, y=0.965,
        text="Bromophenol Blue Tracking Dye Front",
        showarrow=False,
        font=dict(size=9.5, color="#57534E" if is_silver else "#5C6B78", family="IBM Plex Mono, monospace")
    )

    # 4. Well Comb Indentations at top
    for i, lane in enumerate(lanes):
        x_center = (i + 0.75) * lane_width
        well_fill = "#F5F3EF" if is_silver else "#F1F5F9"
        well_border = "#D6D3D1" if is_silver else "#CBD5E1"
        fig.add_shape(
            type="rect",
            x0=x_center - 0.046, y0=-0.03,
            x1=x_center + 0.046, y1=0.025,
            fillcolor=well_fill,
            line=dict(color=well_border, width=1.2),
            layer="below"
        )
        fig.add_annotation(
            x=x_center, y=-0.05,
            text=f"<b>W{i+1}</b>",
            showarrow=False,
            font=dict(size=10, color="#44403C" if is_silver else "#475569", family="IBM Plex Mono, monospace")
        )
        fig.add_annotation(
            x=x_center, y=0.045,
            text=f"<b>{lane['name']}</b>",
            showarrow=False,
            font=dict(size=11, color="#1C1917" if is_silver else "#111827", family="IBM Plex Sans, Helvetica Neue, Arial, sans-serif")
        )

    # Function to draw diffused, realistic bands (multi-pass Gaussian feathering)
    def draw_realistic_band(
        x_center: float,
        rf: float,
        width: float,
        height: float,
        intensity: float,
        label: str,
        mw: float,
        color_type: str = "standard",
        is_prominent: bool = False,
        extra_note: str = ""
    ):
        if is_silver:
            # Metallic silver grains reduced on protein side chains: warm dark sepia / charcoal-brown tones
            if is_prominent:
                c_core = f"rgba(40, 22, 14, {min(0.97, intensity):.2f})"
                c_halo = f"rgba(120, 80, 52, {min(0.45, intensity * 0.45):.2f})"
                b_line = "rgba(25, 12, 8, 0.95)"
            else:
                c_core = f"rgba(58, 36, 24, {min(0.85, intensity):.2f})"
                c_halo = f"rgba(140, 100, 70, {min(0.35, intensity * 0.35):.2f})"
                b_line = "rgba(45, 25, 16, 0.75)"
        else:
            if color_type == "pink_ref":
                c_core = f"rgba(225, 29, 72, {min(0.95, intensity):.2f})"
                c_halo = f"rgba(251, 113, 133, {min(0.40, intensity * 0.4):.2f})"
                b_line = f"rgba(190, 18, 60, {min(1.0, intensity + 0.1):.2f})"
            elif color_type == "green_ref":
                c_core = f"rgba(13, 148, 136, {min(0.95, intensity):.2f})"
                c_halo = f"rgba(45, 212, 191, {min(0.40, intensity * 0.4):.2f})"
                b_line = f"rgba(15, 118, 110, {min(1.0, intensity + 0.1):.2f})"
            elif color_type == "purple_ref":
                c_core = f"rgba(126, 34, 206, {min(0.90, intensity):.2f})"
                c_halo = f"rgba(192, 132, 252, {min(0.38, intensity * 0.38):.2f})"
                b_line = f"rgba(107, 33, 168, {min(1.0, intensity + 0.1):.2f})"
            elif color_type == "yellow_ref":
                c_core = f"rgba(202, 138, 4, {min(0.90, intensity):.2f})"
                c_halo = f"rgba(250, 204, 21, {min(0.38, intensity * 0.38):.2f})"
                b_line = f"rgba(161, 98, 7, {min(1.0, intensity + 0.1):.2f})"
            else:
                # Classic Coomassie Brilliant Blue R-250
                if is_prominent:
                    c_core = f"rgba(30, 64, 175, {min(0.96, intensity):.2f})"
                    c_halo = f"rgba(96, 165, 250, {min(0.45, intensity * 0.45):.2f})"
                    b_line = "rgba(29, 78, 216, 0.90)"
                else:
                    c_core = f"rgba(37, 99, 235, {min(0.85, intensity):.2f})"
                    c_halo = f"rgba(147, 197, 253, {min(0.35, intensity * 0.35):.2f})"
                    b_line = "rgba(30, 64, 175, 0.70)"

        # Layer 1: Soft diffuse halo (Gaussian dispersion)
        fig.add_shape(
            type="rect",
            x0=x_center - (width * 0.52),
            y0=rf - (height * 0.75),
            x1=x_center + (width * 0.52),
            y1=rf + (height * 0.75),
            fillcolor=c_halo,
            line_width=0,
            layer="above"
        )
        # Layer 2: Dense band core
        fig.add_shape(
            type="rect",
            x0=x_center - (width * 0.48),
            y0=rf - (height * 0.45),
            x1=x_center + (width * 0.48),
            y1=rf + (height * 0.45),
            fillcolor=c_core,
            line=dict(color=b_line, width=0.8),
            layer="above"
        )

        # Normalize displayed relative mobility within resolving gel [0 = interface, 1 = dye front]
        rel_rf = float(np.clip((rf - 0.075) / (0.940 - 0.075), 0.01, 1.00))
        hover_desc = f"<b>{label}</b><br>MW: {mw:.1f} kDa<br>Resolving Rf: {rel_rf:.2f}"
        if extra_note:
            hover_desc += f"<br><i>{extra_note}</i>"
        if is_silver and not is_prominent:
            hover_desc += "<br><span style='color:#A16207;'>Silver Stain (~0.2–1.0 ng/band sensitivity)</span>"

        fig.add_trace(go.Scatter(
            x=[x_center],
            y=[rf],
            mode="markers",
            marker=dict(size=15, color="rgba(0,0,0,0)"),
            name=label,
            hoverinfo="text",
            hovertext=hover_desc,
            showlegend=False
        ))

    # 5. Populate bands in each lane
    target_rf = calculate_sds_rf(target_mw_kda, gel_pct)
    dimer_mw = round(target_mw_kda * 2.0, 2)
    dimer_rf = calculate_sds_rf(dimer_mw, gel_pct)

    # Authentic abundant E. coli BL21(DE3) host proteins (>50–100 ng/band -> visible on both Coomassie R-250 & Silver Stain)
    # Membrane porins (OmpF/OmpA ~37.2 kDa) & large RNAP assemblies are enriched in whole-cell Lysate and pelleted/reduced upon clarification
    abundant_host_mws = [
        {"mw": 153.0, "name": "RNAP β/β' (RpoB/C)",     "intensity": 0.34, "h": 0.007, "pellet_only": False},
        {"mw": 99.7,  "name": "Pyruvate DH E1 (AceE)",  "intensity": 0.38, "h": 0.007, "pellet_only": False},
        {"mw": 87.4,  "name": "Lon / ClpB Protease",    "intensity": 0.36, "h": 0.007, "pellet_only": False},
        {"mw": 69.1,  "name": "DnaK (Hsp70 Chaperone)", "intensity": 0.54, "h": 0.009, "pellet_only": False},
        {"mw": 57.3,  "name": "GroEL (Hsp60 Monomer)",  "intensity": 0.58, "h": 0.009, "pellet_only": False},
        {"mw": 48.2,  "name": "Trigger Factor (Tig)",   "intensity": 0.40, "h": 0.008, "pellet_only": False},
        {"mw": 43.3,  "name": "EF-Tu (TufA/B, Major)",  "intensity": 0.62, "h": 0.009, "pellet_only": False},
        {"mw": 37.2,  "name": "OmpF/OmpA Outer Porins", "intensity": 0.50, "h": 0.008, "pellet_only": True},
        {"mw": 35.5,  "name": "GAPDH (GapA)",           "intensity": 0.42, "h": 0.008, "pellet_only": False},
        {"mw": 29.9,  "name": "Ribosomal / Metabolic",  "intensity": 0.35, "h": 0.007, "pellet_only": False},
        {"mw": 23.6,  "name": "CRP / SlyD-band Region", "intensity": 0.36, "h": 0.007, "pellet_only": False},
        {"mw": 15.8,  "name": "Ribosomal L7/L12 / H-NS","intensity": 0.32, "h": 0.006, "pellet_only": False},
    ]

    # Low-abundance E. coli host proteins (~0.2–20 ng/band -> below Coomassie R-250 ~50 ng limit, revealed by Silver Stain!)
    silver_low_abundance_mws = [
        {"mw": 215.0, "intensity": 0.18, "h": 0.005},
        {"mw": 170.0, "intensity": 0.21, "h": 0.005},  # MukB
        {"mw": 132.0, "intensity": 0.23, "h": 0.006},
        {"mw": 118.0, "intensity": 0.20, "h": 0.005},
        {"mw": 93.5,  "intensity": 0.25, "h": 0.006},  # AcnB
        {"mw": 74.3,  "intensity": 0.27, "h": 0.006},  # ArnA (endogenous His-cluster protein)
        {"mw": 63.0,  "intensity": 0.26, "h": 0.006},  # Ribosomal S1 (RpsA)
        {"mw": 52.0,  "intensity": 0.25, "h": 0.006},
        {"mw": 40.5,  "intensity": 0.28, "h": 0.006},  # Pgk
        {"mw": 33.0,  "intensity": 0.24, "h": 0.005},
        {"mw": 26.8,  "intensity": 0.25, "h": 0.005},  # Triosephosphate isomerase (TpiA)
        {"mw": 20.9,  "intensity": 0.26, "h": 0.005},  # SlyD (His-rich FKBP chaperone)
        {"mw": 18.2,  "intensity": 0.22, "h": 0.005},  # AhpC
        {"mw": 13.5,  "intensity": 0.20, "h": 0.005},
        {"mw": 11.2,  "intensity": 0.18, "h": 0.005},
    ]

    for i, lane in enumerate(lanes):
        x_center = (i + 0.75) * lane_width
        b_width = 0.076

        if lane["type"] == "ladder":
            # LAST WELL: Bio-Rad Precision Plus Protein Dual Color Standard
            band_rfs = [calculate_sds_rf(b["mw_kda"], gel_pct) for b in ladder_bands]
            label_ys = _compute_non_overlapping_label_positions(band_rfs, min_gap=0.022)

            for b, rf, lbl_y in zip(ladder_bands, band_rfs, label_ys):
                h = 0.012 if b["prominent"] else 0.009
                inte = 0.95 if b["prominent"] else 0.75
                ref_note = (
                    "Bio-Rad Precision Plus Pink Reference Band (~3x intensity)"
                    if b["color_type"] == "pink_ref"
                    else ("Bio-Rad Precision Plus High-Intensity Blue Reference Band" if b["prominent"] else "Bio-Rad Precision Plus Recombinant Standard")
                )
                draw_realistic_band(
                    x_center, rf, b_width, h, inte,
                    f"Ladder {b['label']}", b["mw_kda"],
                    color_type="standard" if is_silver else b.get("color_type", "blue"),
                    is_prominent=b["prominent"],
                    extra_note=ref_note
                )

                lbl_color = (
                    "#451A03" if (is_silver and b["prominent"])
                    else ("#78350F" if is_silver
                          else ("#BE123C" if b["color_type"] == "pink_ref"
                                else ("#1E3A8A" if b["prominent"] else "#64748B")))
                )
                fig.add_annotation(
                    x=x_center + 0.048,
                    y=lbl_y,
                    text=f"<b>{b['label']}</b>",
                    showarrow=False,
                    font=dict(
                        size=9.5,
                        color=lbl_color,
                        family="IBM Plex Mono, monospace"
                    ),
                    xanchor="left",
                    yanchor="middle"
                )

        elif "Elution" in lane["name"] or "Purified" in lane["name"]:
            # Purified Target Protein (Main Overexpressed Monomer Band)
            draw_realistic_band(
                x_center, target_rf, b_width, 0.015, 0.98,
                f"Target Protein ({target_mw_kda:.1f} kDa)", target_mw_kda,
                color_type="standard", is_prominent=True,
                extra_note="Purified Recombinant Monomer (>95% purity on Coomassie R-250)"
            )

            if include_dimer and dimer_mw <= 600.0:
                draw_realistic_band(
                    x_center, dimer_rf, b_width, 0.007, 0.28,
                    f"Unreduced Disulfide / Crosslinked Dimer ({dimer_mw:.1f} kDa)", dimer_mw,
                    color_type="standard", is_prominent=False,
                    extra_note="Non-covalent dimers dissociate in SDS; visible only if disulfide-bridged without DTT/β-ME"
                )

            # Under ultra-sensitive Silver Stain (~0.2–1.0 ng/band): reveal trace co-eluting E. coli chaperones/contaminants
            if is_silver:
                for trace_mw, trace_lbl, trace_int in [
                    (69.1, "Trace Host Chaperone DnaK (~69.1 kDa)", 0.16),
                    (74.3, "Trace Endogenous His-Cluster ArnA (~74.3 kDa)", 0.13),
                    (20.9, "Trace His-Rich Chaperone SlyD (~20.9 kDa)", 0.14),
                ]:
                    if abs(trace_mw - target_mw_kda) > 2.0:
                        draw_realistic_band(
                            x_center, calculate_sds_rf(trace_mw, gel_pct), b_width, 0.005, trace_int,
                            trace_lbl, trace_mw,
                            extra_note="Sub-nanogram (~0.5–2 ng) trace visible on Silver Stain; invisible on Coomassie R-250 (<50 ng)"
                        )

        elif "Lysate" in lane["name"]:
            # Total Cell Extract / Input (includes insoluble membrane porins OmpF/OmpA):
            for bg in abundant_host_mws:
                if abs(bg["mw"] - target_mw_kda) > 1.2:
                    rf_bg = calculate_sds_rf(bg["mw"], gel_pct)
                    draw_realistic_band(
                        x_center, rf_bg, b_width, bg["h"], bg["intensity"],
                        f"Host {bg['name']} (~{bg['mw']:.1f} kDa)", bg["mw"],
                        extra_note="Abundant E. coli BL21(DE3) whole-cell protein (>50 ng)"
                    )

            if is_silver:
                for bg in silver_low_abundance_mws:
                    if abs(bg["mw"] - target_mw_kda) > 1.0:
                        rf_bg = calculate_sds_rf(bg["mw"], gel_pct)
                        draw_realistic_band(
                            x_center, rf_bg, b_width, bg["h"], bg["intensity"],
                            f"Low-Abundance E. coli Protein (~{bg['mw']:.1f} kDa)", bg["mw"],
                            extra_note="Detected via Silver Stain (~0.2–1.0 ng limit)"
                        )

            draw_realistic_band(
                x_center, target_rf, b_width, 0.014, 0.90,
                f"Induced Target Protein ({target_mw_kda:.1f} kDa)", target_mw_kda,
                is_prominent=True, extra_note="IPTG-Induced Overexpressed Target in Whole-Cell Lysate"
            )

        elif "Soluble" in lane["name"]:
            # Soluble Fraction (Clarified High-Speed Supernatant — outer-membrane OmpF/OmpA pelleted out!):
            for bg in abundant_host_mws:
                if not bg["pellet_only"] and abs(bg["mw"] - target_mw_kda) > 1.2:
                    rf_bg = calculate_sds_rf(bg["mw"], gel_pct)
                    draw_realistic_band(
                        x_center, rf_bg, b_width, bg["h"] * 0.92, bg["intensity"] * 0.88,
                        f"Soluble Host {bg['name']} (~{bg['mw']:.1f} kDa)", bg["mw"],
                        extra_note="Clarified soluble E. coli cytoplasmic fraction"
                    )

            if is_silver:
                for bg in silver_low_abundance_mws[1:]:
                    if abs(bg["mw"] - target_mw_kda) > 1.0:
                        rf_bg = calculate_sds_rf(bg["mw"], gel_pct)
                        draw_realistic_band(
                            x_center, rf_bg, b_width, bg["h"], bg["intensity"] * 0.86,
                            f"Soluble Low-Abundance Protein (~{bg['mw']:.1f} kDa)", bg["mw"],
                            extra_note="Detected via Silver Stain (~0.2–1.0 ng limit)"
                        )

            draw_realistic_band(
                x_center, target_rf, b_width, 0.013, 0.85,
                f"Soluble Target Fraction ({target_mw_kda:.1f} kDa)", target_mw_kda,
                is_prominent=True, extra_note="Soluble Folded Target Loaded onto Column"
            )

        elif "Flowthru" in lane["name"]:
            # Flowthrough (FT / Unbound Column Fraction — target selectively depleted onto resin!):
            for bg in abundant_host_mws:
                if not bg["pellet_only"]:
                    rf_bg = calculate_sds_rf(bg["mw"], gel_pct)
                    draw_realistic_band(
                        x_center, rf_bg, b_width, bg["h"] * 0.90, bg["intensity"] * 0.86,
                        f"Unbound Host {bg['name']} (~{bg['mw']:.1f} kDa)", bg["mw"],
                        extra_note="Non-binding host protein washing into Flowthrough"
                    )

            if is_silver:
                for bg in silver_low_abundance_mws[1:]:
                    rf_bg = calculate_sds_rf(bg["mw"], gel_pct)
                    draw_realistic_band(
                        x_center, rf_bg, b_width, bg["h"], bg["intensity"] * 0.84,
                        f"Unbound Low-Abundance Protein (~{bg['mw']:.1f} kDa)", bg["mw"],
                        extra_note="Unbound host fraction (Silver-detected)"
                    )

            # Minor unbound target trace (dynamic binding capacity breakthrough)
            draw_realistic_band(
                x_center, target_rf, b_width, 0.006, 0.14 if not is_silver else 0.28,
                f"Unbound Target Trace ({target_mw_kda:.1f} kDa)", target_mw_kda,
                is_prominent=False,
                extra_note="Minor column breakthrough (<5% unbound target)"
            )

    fig.update_xaxes(
        range=[-0.04, 1.15],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )

    # Invert Y so electrophoretic migration flows top (-) -> bottom (+)
    fig.update_yaxes(
        range=[1.02, -0.09],
        showgrid=False,
        zeroline=False,
        showticklabels=False
    )

    stain_sub = (
        "Silver Stain (~0.2–1.0 ng sensitivity)"
        if is_silver
        else "Coomassie R-250 (~50–100 ng sensitivity)"
    )
    fig.update_layout(
        font=dict(family="IBM Plex Sans, Helvetica Neue, Arial, sans-serif", color="#2C3A47", size=12),
        title=dict(
            text=(
                f"<b>Simulated Destained SDS-PAGE Gel ({gel_pct:.1f}% Polyacrylamide)</b><br>"
                f"<span style='font-size:12px; color:#52606D;'>Target MW: {target_mw_kda:.1f} kDa | "
                f"Bio-Rad Precision Plus Dual Color in Last Well (Lane {num_lanes}) | {stain_sub}</span>"
            ),
            font=dict(size=14, color="#111827", family="IBM Plex Sans, Helvetica Neue, Arial, sans-serif")
        ),
        template="plotly_white",
        height=550,
        margin=dict(l=20, r=32, t=68, b=25),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="#FFFFFF"
    )

    return fig
