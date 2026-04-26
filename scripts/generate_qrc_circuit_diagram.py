#!/usr/bin/env python3
"""
generate_qrc_circuit_diagram.py — Draw the QRC circuit and save as PNG.

The QRC circuit structure (from src/quantum_reservoir.py):
  1. Input encoding on 5 qubits (RY rotations)
     - q0: RY(2π(x+20)/40)   — x component
     - q1: RY(2π(y+30)/60)   — y component
     - q2: RY(2π·z/50)        — z component
     - q3: RY((θ_x+θ_y)/2)   — cross term
     - q4: RY((θ_y+θ_z)/2)   — cross term
  2. For each of 2 reservoir layers (fixed random parameters):
     - Single-qubit RX, RY, RZ on each qubit
     - CNOT ring: q0→q1, q1→q2, q2→q3, q3→q4
     - After each CNOT: RZ(coupling_strength * random_angle) on target
     - Ring closure: q4→q0 + RZ on q0
  3. Measure all qubits → 32-dimensional probability vector

Uses PennyLane for drawing (matplotlib backend).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ── Matplotlib-only circuit diagram (no PennyLane dependency needed) ──────────

def draw_qrc_circuit(output_path: str, dpi: int = 300):
    """
    Draw the QRC circuit diagram using matplotlib directly.

    Layout:
      - 5 horizontal qubit wires
      - Input encoding column: RY gates
      - Layer 1: RX/RY/RZ column + CNOT ring column
      - Layer 2: RX/RY/RZ column + CNOT ring column
      - Measurement column
    """

    n_qubits = 5
    qubit_labels = [r"$q_0$", r"$q_1$", r"$q_2$", r"$q_3$", r"$q_4$"]

    # Column x-positions
    x_start     = 0.5
    x_enc       = 1.5          # Input encoding RY
    x_l1_rot    = 3.0          # Layer 1: single-qubit rotations
    x_l1_ent    = 4.5          # Layer 1: entangling CNOTs
    x_l2_rot    = 6.0          # Layer 2: single-qubit rotations
    x_l2_ent    = 7.5          # Layer 2: entangling CNOTs
    x_meas      = 9.0          # Measurement
    x_end       = 10.0

    y_wire = [n_qubits - i for i in range(n_qubits)]  # y=5,4,3,2,1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, x_end + 0.5)
    ax.set_ylim(0, n_qubits + 1)
    ax.axis("off")

    gate_w, gate_h = 0.55, 0.4

    def draw_wire(ax, y, x0, x1, color="black", lw=1.2):
        ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=1)

    def draw_gate(ax, x, y, label, color="#AED6F1", fontsize=7):
        box = FancyBboxPatch(
            (x - gate_w/2, y - gate_h/2), gate_w, gate_h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="black", linewidth=1.0, zorder=3
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                fontweight="bold", zorder=4)

    def draw_cnot(ax, x, y_ctrl, y_tgt, color="black"):
        # Control dot
        ax.plot(x, y_ctrl, "o", markersize=6, color=color, zorder=4)
        # Line
        ax.plot([x, x], [y_ctrl, y_tgt], color=color, lw=1.2, zorder=3)
        # Target circle with +
        circle = plt.Circle((x, y_tgt), 0.18, fill=False,
                             edgecolor=color, linewidth=1.2, zorder=4)
        ax.add_patch(circle)
        ax.plot([x - 0.18, x + 0.18], [y_tgt, y_tgt], color=color, lw=1.2, zorder=5)
        ax.plot([x, x], [y_tgt - 0.18, y_tgt + 0.18], color=color, lw=1.2, zorder=5)

    def draw_measure(ax, x, y):
        box = FancyBboxPatch(
            (x - gate_w/2, y - gate_h/2), gate_w, gate_h,
            boxstyle="round,pad=0.04",
            facecolor="#F9E79F", edgecolor="black", linewidth=1.0, zorder=3
        )
        ax.add_patch(box)
        # Meter symbol
        ax.text(x, y, "M", ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=4)

    # ── Draw wires ──────────────────────────────────────────────────────────
    for i, y in enumerate(y_wire):
        draw_wire(ax, y, x_start, x_end)
        ax.text(x_start - 0.3, y, qubit_labels[i], ha="right", va="center",
                fontsize=9, fontweight="bold")

    # ── Input encoding labels ──────────────────────────────────────────────
    enc_labels = [
        r"$R_Y(\theta_x)$",
        r"$R_Y(\theta_y)$",
        r"$R_Y(\theta_z)$",
        r"$R_Y(\frac{\theta_x+\theta_y}{2})$",
        r"$R_Y(\frac{\theta_y+\theta_z}{2})$",
    ]
    enc_colors = ["#A9DFBF"] * 5  # green for encoding

    for i, (y, lbl, col) in enumerate(zip(y_wire, enc_labels, enc_colors)):
        fs = 5.5 if i >= 3 else 6.5
        draw_gate(ax, x_enc, y, lbl, color=col, fontsize=fs)

    # Column header
    ax.text(x_enc, n_qubits + 0.5, "Input\nEncoding",
            ha="center", va="center", fontsize=7, color="#1A5276",
            fontweight="bold")

    # ── Reservoir layers ─────────────────────────────────────────────────────
    rot_color = "#AED6F1"  # blue for reservoir rotations
    ent_color = "#E8DAEF"  # purple for entangling

    for layer_idx, (x_rot, x_ent) in enumerate(
        [(x_l1_rot, x_l1_ent), (x_l2_rot, x_l2_ent)]
    ):
        layer_num = layer_idx + 1
        ax.text(x_rot, n_qubits + 0.5, f"Layer {layer_num}\nRotations",
                ha="center", va="center", fontsize=7, color="#1A5276",
                fontweight="bold")
        ax.text(x_ent, n_qubits + 0.5, f"Layer {layer_num}\nEntangling",
                ha="center", va="center", fontsize=7, color="#6C3483",
                fontweight="bold")

        # Single-qubit rotation block per qubit
        for y in y_wire:
            draw_gate(ax, x_rot, y, r"$R_X R_Y R_Z$",
                      color=rot_color, fontsize=5.5)

        # CNOT ring: q0→q1, q1→q2, q2→q3, q3→q4
        for ctrl in range(n_qubits - 1):
            draw_cnot(ax, x_ent, y_wire[ctrl], y_wire[ctrl + 1])

        # Ring closure: q4→q0 (drawn as a curved arc on the left of the column)
        y_last  = y_wire[-1]  # q4 (y=1)
        y_first = y_wire[0]   # q0 (y=5)
        arc_x   = x_ent + 0.4
        arc = matplotlib.patches.FancyArrowPatch(
            (arc_x, y_last), (arc_x, y_first),
            connectionstyle="arc3,rad=-0.4",
            arrowstyle="-|>",
            color="#6C3483", linewidth=1.0, zorder=3,
            mutation_scale=8
        )
        ax.add_patch(arc)
        ax.text(arc_x + 0.25, (y_last + y_first) / 2, "ring",
                fontsize=5, color="#6C3483", va="center")

    # ── Measurement ──────────────────────────────────────────────────────────
    for y in y_wire:
        draw_measure(ax, x_meas, y)

    ax.text(x_meas, n_qubits + 0.5, "Measure\n(shots)",
            ha="center", va="center", fontsize=7, color="#7D6608",
            fontweight="bold")

    # ── Output annotation ─────────────────────────────────────────────────
    ax.annotate(
        r"$\mathbf{f}\in\mathbb{R}^{32}$" + "\n(prob. vector)",
        xy=(x_end, (y_wire[0] + y_wire[-1]) / 2),
        xytext=(x_end + 0.05, (y_wire[0] + y_wire[-1]) / 2),
        fontsize=7, va="center", color="#7D6608"
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#A9DFBF", edgecolor="black", label="Input encoding (trainable-free)"),
        mpatches.Patch(facecolor=rot_color, edgecolor="black", label=r"Fixed random $R_X R_Y R_Z$ (frozen)"),
        mpatches.Patch(facecolor=ent_color, edgecolor="black", label="CNOT ring + RZ (frozen)"),
        mpatches.Patch(facecolor="#F9E79F", edgecolor="black", label="Measurement"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=6.5,
              framealpha=0.9)

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        "QRC Circuit: 5 qubits, 2 fixed reservoir layers, ring-topology entanglement",
        fontsize=9, pad=6
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Circuit diagram saved to: {output_path}")


if __name__ == "__main__":
    out = Path(__file__).parent.parent / "paper" / "fig2_circuit.png"
    draw_qrc_circuit(str(out), dpi=300)
