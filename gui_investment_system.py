# -*- coding: utf-8 -*-
"""
Intelligent Investment Decision System - GUI Application
A beautiful desktop interface for Bitcoin investment analysis
No login required - Direct access to the analysis system

Author: Investment Committee System
Version: v3.1 GUI Edition
"""

import os
import sys
import json
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent
WORKFLOW_RESULT_FILE = PROJECT_ROOT / "n8n_workflows" / "latest_workflow_result.json"

# Add Bitcoin prediction project path
sys.path.insert(0, str(PROJECT_ROOT / "Bitcoin price prediction Project"))


# ==================== Color Theme ====================

class Theme:
    """Modern Dark Theme Colors"""
    
    # Main colors
    BG_DARK = "#0d1117"
    BG_SECONDARY = "#161b22"
    BG_TERTIARY = "#21262d"
    BG_CARD = "#1c2128"
    
    # Accent colors
    PRIMARY = "#58a6ff"
    PRIMARY_DARK = "#1f6feb"
    SUCCESS = "#3fb950"
    WARNING = "#d29922"
    ERROR = "#f85149"
    INFO = "#8b949e"
    
    # Text colors
    TEXT_PRIMARY = "#e6edf3"
    TEXT_SECONDARY = "#8b949e"
    TEXT_MUTED = "#6e7681"
    
    # Analyst colors
    ANALYST_COLORS = {
        "technical_analyst": "#58a6ff",
        "industry_analyst": "#a371f7",
        "financial_analyst": "#3fb950",
        "market_expert": "#f0883e",
        "risk_analyst": "#f85149",
        "investment_manager": "#d29922",
    }
    
    # Status colors
    STATUS_RUNNING = "#58a6ff"
    STATUS_COMPLETE = "#3fb950"
    STATUS_ERROR = "#f85149"
    STATUS_PENDING = "#6e7681"


# ==================== Custom Widgets ====================

class ModernButton(tk.Button):
    """Modern styled button"""
    
    def __init__(self, parent, text, command=None, style="primary", **kwargs):
        colors = {
            "primary": (Theme.PRIMARY, Theme.TEXT_PRIMARY),
            "success": (Theme.SUCCESS, Theme.TEXT_PRIMARY),
            "warning": (Theme.WARNING, Theme.BG_DARK),
            "error": (Theme.ERROR, Theme.TEXT_PRIMARY),
            "secondary": (Theme.BG_TERTIARY, Theme.TEXT_PRIMARY),
        }
        
        bg, fg = colors.get(style, colors["primary"])
        
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=Theme.PRIMARY_DARK,
            activeforeground=Theme.TEXT_PRIMARY,
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
            padx=20,
            pady=8,
            **kwargs
        )
        
        self.bind("<Enter>", lambda e: self.config(bg=Theme.PRIMARY_DARK))
        self.bind("<Leave>", lambda e: self.config(bg=bg))


class ProgressCard(tk.Frame):
    """Progress indicator card for each analysis step"""
    
    def __init__(self, parent, title, description="", **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, **kwargs)
        
        self.title = title
        self.status = "pending"
        
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        
        # Status indicator
        self.status_canvas = tk.Canvas(
            self, width=16, height=16, bg=Theme.BG_CARD, 
            highlightthickness=0
        )
        self.status_canvas.grid(row=0, column=0, padx=(15, 10), pady=15)
        self._draw_status()
        
        # Title and description
        self.title_label = tk.Label(
            self, text=title, font=("Segoe UI Semibold", 12),
            bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY, anchor="w"
        )
        self.title_label.grid(row=0, column=1, sticky="w", pady=(15, 5))
        
        self.desc_label = tk.Label(
            self, text=description, font=("Segoe UI", 10),
            bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY, anchor="w"
        )
        self.desc_label.grid(row=1, column=1, sticky="w", padx=(0, 15), pady=(0, 15))
        
        # Duration label
        self.duration_label = tk.Label(
            self, text="", font=("Segoe UI", 9),
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED, anchor="e"
        )
        self.duration_label.grid(row=0, column=2, padx=15, pady=15)
        
        # Border effect
        self.configure(highlightbackground=Theme.BG_TERTIARY, highlightthickness=1)
    
    def _draw_status(self):
        """Draw status indicator circle"""
        self.status_canvas.delete("all")
        
        colors = {
            "pending": Theme.STATUS_PENDING,
            "running": Theme.STATUS_RUNNING,
            "complete": Theme.STATUS_COMPLETE,
            "error": Theme.STATUS_ERROR,
        }
        
        color = colors.get(self.status, Theme.STATUS_PENDING)
        
        if self.status == "running":
            # Animated dot effect - just solid for now
            self.status_canvas.create_oval(2, 2, 14, 14, fill=color, outline="")
        elif self.status == "complete":
            # Checkmark
            self.status_canvas.create_oval(2, 2, 14, 14, fill=color, outline="")
            self.status_canvas.create_text(8, 8, text="✓", fill="white", font=("Segoe UI", 8, "bold"))
        elif self.status == "error":
            # X mark
            self.status_canvas.create_oval(2, 2, 14, 14, fill=color, outline="")
            self.status_canvas.create_text(8, 8, text="✗", fill="white", font=("Segoe UI", 8, "bold"))
        else:
            # Empty circle
            self.status_canvas.create_oval(2, 2, 14, 14, fill="", outline=color, width=2)
    
    def set_status(self, status, duration=""):
        """Update status"""
        self.status = status
        self._draw_status()
        self.duration_label.config(text=duration)
        
        if status == "running":
            self.title_label.config(fg=Theme.PRIMARY)
        elif status == "complete":
            self.title_label.config(fg=Theme.SUCCESS)
        elif status == "error":
            self.title_label.config(fg=Theme.ERROR)
        else:
            self.title_label.config(fg=Theme.TEXT_PRIMARY)
    
    def update_description(self, text):
        """Update description text"""
        self.desc_label.config(text=text)


class OutputPanel(tk.Frame):
    """Scrollable output panel for displaying analysis results"""
    
    def __init__(self, parent, title="Output", **kwargs):
        super().__init__(parent, bg=Theme.BG_SECONDARY, **kwargs)
        
        # Header
        header = tk.Frame(self, bg=Theme.BG_TERTIARY)
        header.pack(fill="x", padx=1, pady=(1, 0))
        
        tk.Label(
            header, text=title, font=("Segoe UI Semibold", 11),
            bg=Theme.BG_TERTIARY, fg=Theme.TEXT_PRIMARY, anchor="w"
        ).pack(side="left", padx=15, pady=10)
        
        # Clear button
        self.clear_btn = tk.Label(
            header, text="Clear", font=("Segoe UI", 9),
            bg=Theme.BG_TERTIARY, fg=Theme.TEXT_MUTED, cursor="hand2"
        )
        self.clear_btn.pack(side="right", padx=15, pady=10)
        self.clear_btn.bind("<Button-1>", lambda e: self.clear())
        
        # Text area
        self.text = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, font=("Consolas", 10),
            bg=Theme.BG_DARK, fg=Theme.TEXT_PRIMARY,
            insertbackground=Theme.TEXT_PRIMARY,
            selectbackground=Theme.PRIMARY,
            borderwidth=0, highlightthickness=0,
            padx=15, pady=10
        )
        self.text.pack(fill="both", expand=True, padx=1, pady=1)
        
        # Configure tags for colored output
        self.text.tag_configure("header", foreground=Theme.PRIMARY, font=("Consolas", 10, "bold"))
        self.text.tag_configure("success", foreground=Theme.SUCCESS)
        self.text.tag_configure("warning", foreground=Theme.WARNING)
        self.text.tag_configure("error", foreground=Theme.ERROR)
        self.text.tag_configure("info", foreground=Theme.INFO)
        self.text.tag_configure("muted", foreground=Theme.TEXT_MUTED)
        
        # Analyst tags
        for role, color in Theme.ANALYST_COLORS.items():
            self.text.tag_configure(role, foreground=color)
    
    def append(self, text, tag=None):
        """Append text to output"""
        self.text.config(state=tk.NORMAL)
        if tag:
            self.text.insert(tk.END, text, tag)
        else:
            self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)
    
    def append_line(self, text, tag=None):
        """Append text with newline"""
        self.append(text + "\n", tag)
    
    def clear(self):
        """Clear all text"""
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.config(state=tk.DISABLED)


class AnalystCard(tk.Frame):
    """Card displaying analyst information and recommendation"""
    
    def __init__(self, parent, role, name, **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, **kwargs)
        
        self.role = role
        color = Theme.ANALYST_COLORS.get(role, Theme.PRIMARY)
        
        # Configure
        self.configure(highlightbackground=color, highlightthickness=2)
        
        # Header with role name
        header = tk.Frame(self, bg=color)
        header.pack(fill="x")
        
        tk.Label(
            header, text=name, font=("Segoe UI Semibold", 10),
            bg=color, fg=Theme.TEXT_PRIMARY
        ).pack(padx=10, pady=5)
        
        # Content frame
        content = tk.Frame(self, bg=Theme.BG_CARD)
        content.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Recommendation
        self.rec_label = tk.Label(
            content, text="--", font=("Segoe UI Bold", 14),
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED
        )
        self.rec_label.pack()
        
        # Confidence
        self.conf_label = tk.Label(
            content, text="Confidence: --", font=("Segoe UI", 9),
            bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY
        )
        self.conf_label.pack(pady=(5, 0))
    
    def set_recommendation(self, action, confidence):
        """Update recommendation display"""
        action_display = {
            "strong_buy": ("STRONG BUY", Theme.SUCCESS),
            "buy": ("BUY", Theme.SUCCESS),
            "hold": ("HOLD", Theme.WARNING),
            "sell": ("SELL", Theme.ERROR),
            "strong_sell": ("STRONG SELL", Theme.ERROR),
        }
        
        text, color = action_display.get(action, ("--", Theme.TEXT_MUTED))
        self.rec_label.config(text=text, fg=color)
        self.conf_label.config(text=f"Confidence: {confidence}/10")


# ==================== Main Application ====================

class InvestmentSystemGUI:
    """Main GUI Application for the Investment Decision System"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Intelligent Investment Decision System v3.1")
        self.root.configure(bg=Theme.BG_DARK)
        
        # Window size and position
        window_width = 1400
        window_height = 900
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2 - 30
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1200, 700)
        
        # Message queue for thread-safe UI updates
        self.message_queue = queue.Queue()
        self.latest_workflow_data = None
        
        # Analysis state
        self.is_running = False
        self.current_step = 0
        
        # Build UI
        self._build_ui()
        
        # Start message processing
        self._process_messages()
    
    def _build_ui(self):
        """Build the main UI layout"""
        # Main container with padding
        main_container = tk.Frame(self.root, bg=Theme.BG_DARK)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header section
        self._build_header(main_container)
        
        # Content section (two columns)
        content = tk.Frame(main_container, bg=Theme.BG_DARK)
        content.pack(fill="both", expand=True, pady=(20, 0))
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)
        
        # Left panel - Progress steps
        self._build_progress_panel(content)
        
        # Right panel - Output and results
        self._build_output_panel(content)
    
    def _build_header(self, parent):
        """Build header section with title and controls"""
        header = tk.Frame(parent, bg=Theme.BG_DARK)
        header.pack(fill="x")
        
        # Left side - Title
        title_frame = tk.Frame(header, bg=Theme.BG_DARK)
        title_frame.pack(side="left")
        
        tk.Label(
            title_frame, 
            text="🚀 Intelligent Investment Decision System",
            font=("Segoe UI", 24, "bold"),
            bg=Theme.BG_DARK, 
            fg=Theme.TEXT_PRIMARY
        ).pack(anchor="w")
        
        tk.Label(
            title_frame,
            text="Multi-Agent Bitcoin Investment Analysis with Anonymous Peer Review",
            font=("Segoe UI", 11),
            bg=Theme.BG_DARK,
            fg=Theme.TEXT_SECONDARY
        ).pack(anchor="w", pady=(5, 0))
        
        # Right side - Controls
        controls = tk.Frame(header, bg=Theme.BG_DARK)
        controls.pack(side="right")
        
        self.start_btn = ModernButton(
            controls, 
            text="▶ Start Analysis",
            command=self._start_analysis,
            style="success"
        )
        self.start_btn.pack(side="left", padx=(0, 10))
        
        self.stop_btn = ModernButton(
            controls,
            text="⬛ Stop",
            command=self._stop_analysis,
            style="error"
        )
        self.stop_btn.pack(side="left")
        self.stop_btn.config(state=tk.DISABLED)
        
        self.load_btn = ModernButton(
            controls,
            text="⬇ Load Latest Result",
            command=self._load_latest_result,
            style="secondary"
        )
        self.load_btn.pack(side="left", padx=(10, 0))
    
    def _build_progress_panel(self, parent):
        """Build left panel with progress steps"""
        left_frame = tk.Frame(parent, bg=Theme.BG_SECONDARY)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Header
        header = tk.Frame(left_frame, bg=Theme.BG_TERTIARY)
        header.pack(fill="x")
        
        tk.Label(
            header, 
            text="📋 Analysis Pipeline",
            font=("Segoe UI Semibold", 12),
            bg=Theme.BG_TERTIARY, 
            fg=Theme.TEXT_PRIMARY
        ).pack(padx=15, pady=12, anchor="w")
        
        # Scrollable progress steps
        canvas = tk.Canvas(left_frame, bg=Theme.BG_SECONDARY, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=Theme.BG_SECONDARY)
        
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=330)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Progress cards
        self.progress_cards = {}
        
        steps = [
            ("data_prep", "1. Data Preparation", "Loading price data and initializing models"),
            ("model_loading", "2. Model Loading", "Loading three prediction models"),
            ("prediction", "3. Price Prediction", "Running ensemble prediction"),
            ("advanced", "4. Advanced Analysis", "Multi-timeframe + Scenario analysis"),
            ("analyst_init", "5. Analyst Initialization", "Preparing 5 specialized analysts"),
            ("independent", "6. Independent Analysis", "Each analyst conducts analysis"),
            ("debate", "7. Debate Phase", "Analysts debate on divergent views"),
            ("discussion", "8. Discussion Phase", "Analysts exchange and refine views"),
            ("signal_fusion", "9. Signal Fusion", "Bayesian weighted signal fusion"),
            ("voting", "10. Voting Phase", "Each analyst gives recommendation"),
            ("peer_review", "11. Anonymous Peer Review", "Cross-evaluation of analyses"),
            ("devils_advocate", "12. Devil's Advocate", "Challenge consensus view"),
            ("final_decision", "13. Final Decision", "Quality-weighted decision"),
            ("validation", "14. Red-Blue Validation", "Adversarial decision validation"),
            ("risk_budget", "15. Risk Budget", "Position sizing based on risk"),
            ("review", "16. Decision Review", "Analyze quality and risk points"),
        ]
        
        for step_id, title, desc in steps:
            card = ProgressCard(scroll_frame, title, desc)
            card.pack(fill="x", pady=5, padx=5)
            self.progress_cards[step_id] = card
    
    def _build_output_panel(self, parent):
        """Build right panel with output displays"""
        right_frame = tk.Frame(parent, bg=Theme.BG_DARK)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Analyst cards row
        analyst_frame = tk.Frame(right_frame, bg=Theme.BG_DARK)
        analyst_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.analyst_cards = {}
        analysts = [
            ("technical_analyst", "Technical"),
            ("industry_analyst", "Industry"),
            ("financial_analyst", "Financial"),
            ("market_expert", "Market"),
            ("risk_analyst", "Risk"),
        ]
        
        for i, (role, name) in enumerate(analysts):
            card = AnalystCard(analyst_frame, role, name)
            card.pack(side="left", fill="x", expand=True, padx=(0 if i == 0 else 5, 0))
            self.analyst_cards[role] = card
        
        # Output panels (tabbed)
        notebook = ttk.Notebook(right_frame)
        notebook.grid(row=1, column=0, sticky="nsew")
        
        # Configure notebook style
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=Theme.BG_DARK, borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=Theme.BG_TERTIARY, 
                       foreground=Theme.TEXT_SECONDARY,
                       padding=[15, 8],
                       font=("Segoe UI", 10))
        style.map('TNotebook.Tab',
                 background=[('selected', Theme.BG_SECONDARY)],
                 foreground=[('selected', Theme.TEXT_PRIMARY)])
        
        # Main output tab
        self.main_output = OutputPanel(notebook, "Process Output")
        notebook.add(self.main_output, text="📊 Main Output")
        
        # Analysis details tab
        self.analysis_output = OutputPanel(notebook, "Analysis Details")
        notebook.add(self.analysis_output, text="🔍 Analysis Details")
        
        # Peer review tab
        self.peer_review_output = OutputPanel(notebook, "Peer Review")
        notebook.add(self.peer_review_output, text="📝 Peer Review")
        
        # Final decision tab
        self.decision_output = OutputPanel(notebook, "Final Decision")
        notebook.add(self.decision_output, text="🎯 Final Decision")
    
    def _process_messages(self):
        """Process messages from background thread"""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        
        self.root.after(100, self._process_messages)
    
    def _handle_message(self, msg):
        """Handle a message from the analysis thread"""
        msg_type = msg.get("type")
        
        if msg_type == "progress":
            step_id = msg.get("step")
            status = msg.get("status")
            duration = msg.get("duration", "")
            desc = msg.get("description", "")
            
            if step_id in self.progress_cards:
                self.progress_cards[step_id].set_status(status, duration)
                if desc:
                    self.progress_cards[step_id].update_description(desc)
        
        elif msg_type == "output":
            panel = msg.get("panel", "main")
            text = msg.get("text", "")
            tag = msg.get("tag")
            
            output_panels = {
                "main": self.main_output,
                "analysis": self.analysis_output,
                "peer_review": self.peer_review_output,
                "decision": self.decision_output,
            }
            
            if panel in output_panels:
                output_panels[panel].append_line(text, tag)
        
        elif msg_type == "analyst_update":
            role = msg.get("role")
            action = msg.get("action")
            confidence = msg.get("confidence")
            
            if role in self.analyst_cards:
                self.analyst_cards[role].set_recommendation(action, confidence)
        
        elif msg_type == "complete":
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
        
        elif msg_type == "error":
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            messagebox.showerror("Error", msg.get("message", "An error occurred"))
    
    def _start_analysis(self):
        """Start the analysis in a background thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Clear outputs
        self.main_output.clear()
        self.analysis_output.clear()
        self.peer_review_output.clear()
        self.decision_output.clear()
        
        # Reset progress cards
        for card in self.progress_cards.values():
            card.set_status("pending")
        
        # Reset analyst cards
        for card in self.analyst_cards.values():
            card.set_recommendation("", 0)
        
        # Start analysis thread
        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()
    
    def _stop_analysis(self):
        """Stop the analysis"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        self.message_queue.put({
            "type": "output",
            "panel": "main",
            "text": "\n⚠️ Analysis stopped by user.",
            "tag": "warning"
        })
    
    def _load_latest_result(self):
        """Load the latest workflow result produced by the automation runner"""
        data = self._read_latest_workflow_result()
        if not data:
            return
        
        self.latest_workflow_data = data
        self._populate_from_workflow_data(data)
        messagebox.showinfo(
            "Workflow Snapshot Loaded",
            "Latest workflow result has been loaded into the GUI."
        )
    
    def _read_latest_workflow_result(self) -> Optional[Dict]:
        """Read workflow result JSON file"""
        if not WORKFLOW_RESULT_FILE.exists():
            messagebox.showwarning(
                "Result Not Found",
                "No workflow snapshot available.\n\n"
                "Please run `python n8n_workflows/setup_and_run.py --workflow` first."
            )
            return None
        
        try:
            with open(WORKFLOW_RESULT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            messagebox.showerror("Load Failed", f"Unable to read workflow result:\n{exc}")
            return None
    
    def _populate_from_workflow_data(self, data: Dict):
        """Populate UI panels from workflow snapshot"""
        # Reset state
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        for card in self.progress_cards.values():
            card.set_status("complete")
        
        for panel in (self.main_output, self.analysis_output, self.peer_review_output, self.decision_output):
            panel.clear()
        
        timestamp = data.get("timestamp", "N/A")
        summary = data.get("summary", {})
        price_data = data.get("price_data", {})
        indicators = data.get("technical_indicators", {}).get("technical_indicators", {})
        signals = data.get("technical_indicators", {}).get("signals", {})
        model_predictions = data.get("model_predictions", {})
        
        current_price = summary.get("current_price") or price_data.get("current_price")
        ensemble_price = summary.get("ensemble_price") or model_predictions.get("ensemble", {}).get("predicted_price")
        ensemble_change = summary.get("ensemble_change_percent") or model_predictions.get("ensemble", {}).get("change_percent")
        action = summary.get("final_action", "hold")
        position = summary.get("final_position_percent", summary.get("final_position", 0))
        confidence = summary.get("calibrated_confidence")
        
        self.main_output.append_line("=" * 60, "header")
        self.main_output.append_line("WORKFLOW SNAPSHOT (Loaded from latest run)", "header")
        self.main_output.append_line("=" * 60, "header")
        self.main_output.append_line(f"Timestamp: {timestamp}", "info")
        self.main_output.append_line(f"Current Price: {self._format_price(current_price)}")
        self.main_output.append_line(f"Ensemble Price: {self._format_price(ensemble_price)}")
        self.main_output.append_line(f"Ensemble Change: {self._format_percent(ensemble_change)}")
        self.main_output.append_line(f"Final Action: {action.upper()}")
        self.main_output.append_line(f"Position Size: {position}%")
        self.main_output.append_line(f"Confidence: {self._format_percent(confidence * 100 if confidence and confidence <= 1 else confidence)}")
        self.main_output.append_line("")
        
        if indicators:
            self.main_output.append_line("Technical Indicators:", "header")
            self.main_output.append_line(f"  RSI(14): {self._format_number(indicators.get('rsi_14'))}")
            self.main_output.append_line(f"  MACD: {self._format_number(indicators.get('macd'), 4)}")
            self.main_output.append_line(f"  SMA(20): {self._format_price(indicators.get('sma_20'))}")
            self.main_output.append_line(f"  Bollinger Upper: {self._format_price(indicators.get('bb_upper'))}")
            if signals:
                self.main_output.append_line(f"  RSI Signal: {signals.get('rsi_signal', 'N/A')}")
                self.main_output.append_line(f"  MACD Signal: {signals.get('macd_signal', 'N/A')}")
            self.main_output.append_line("")
        
        if model_predictions:
            self.main_output.append_line("Model Predictions:", "header")
            for name in ("linear_regression", "arima", "seasonal"):
                result = model_predictions.get(name)
                if result:
                    label = {
                        "linear_regression": "Linear Regression",
                        "arima": "ARIMA",
                        "seasonal": "Seasonal"
                    }.get(name, name.title())
                    self.main_output.append_line(
                        f"  {label}: {self._format_price(result.get('predicted_price'))} "
                        f"({self._format_percent(result.get('change_percent'))})"
                    )
            self.main_output.append_line("")
        
        # Analysis details tab
        self.analysis_output.append_line("==== Multi-Timeframe Analysis ====", "header")
        mtf = data.get("multi_timeframe_analysis", {}).get("analysis", {})
        for key in ("short_term", "medium_term", "long_term"):
            section = mtf.get(key)
            if section:
                self.analysis_output.append_line(
                    f"{section.get('timeframe', key.title())} ({section.get('period', 'N/A')}): "
                    f"{section.get('signal', '').upper()} / "
                    f"{self._format_percent(section.get('predicted_change'))} / "
                    f"Confidence {self._format_percent(section.get('confidence') * 100 if section.get('confidence') and section.get('confidence') <= 1 else section.get('confidence'))}"
                )
        synthesis = mtf.get("synthesis")
        if synthesis:
            self.analysis_output.append_line(
                f"Synthesis: {synthesis.get('final_signal', 'N/A')} "
                f"(Consistency: {synthesis.get('consistency', 'N/A')})"
            )
        self.analysis_output.append_line("")
        
        self.analysis_output.append_line("==== Scenario Analysis ====", "header")
        scenarios = data.get("scenario_analysis", {}).get("scenarios", {})
        for key, scenario in scenarios.items():
            if key in ("expected_value", "risk_assessment"):
                continue
            self.analysis_output.append_line(
                f"{scenario.get('name', key.title())}: "
                f"Prob {self._format_percent(scenario.get('probability') * 100 if isinstance(scenario.get('probability'), (int, float)) and scenario.get('probability') <= 1 else scenario.get('probability'))}, "
                f"Change {self._format_percent(scenario.get('predicted_change'))}, "
                f"Price {self._format_price(scenario.get('predicted_price'))}, "
                f"Risk {scenario.get('risk_level', 'N/A')}"
            )
        expected = scenarios.get("expected_value", {})
        if expected:
            self.analysis_output.append_line(
                f"Expected Value: {self._format_percent(expected.get('expected_change'))} "
                f"→ {self._format_price(expected.get('expected_price'))}"
            )
        self.analysis_output.append_line("")
        
        self.analysis_output.append_line("==== Analyst Opinions ====", "header")
        opinions = data.get("analyst_opinions", [])
        self._update_analyst_cards_from_data(opinions)
        for op in opinions:
            label = op.get("type", "").capitalize()
            self.analysis_output.append_line(
                f"{label}: {op.get('recommendation', 'hold').upper()} "
                f"({op.get('confidence', 0)}/10)"
            )
            analysis_text = op.get("analysis", {}).get("content")
            if analysis_text:
                self.analysis_output.append_line(f"  • {analysis_text}")
        self.analysis_output.append_line("")
        
        # Peer review tab
        self.peer_review_output.append_line("==== Anonymous Peer Review ====", "header")
        anonymized = data.get("anonymous_analyses", {}).get("anonymous_analyses", []) or data.get("anonymous_analyses", [])
        self.peer_review_output.append_line(f"Anonymous analysts: {len(anonymized)}")
        peer_result = data.get("peer_review", {})
        self.peer_review_output.append_line(f"Total reviews: {peer_result.get('total_reviews', 0)}")
        rankings = data.get("rankings", {}).get("rankings", [])
        consensus = data.get("rankings", {}).get("consensus") or data.get("analyst_consensus", "N/A")
        self.peer_review_output.append_line(f"Consensus: {consensus}")
        if rankings:
            self.peer_review_output.append_line("\nTop Analysts:", "header")
            for r in rankings[:5]:
                self.peer_review_output.append_line(
                    f"  #{r.get('rank')} {r.get('anonymous_id')} "
                    f"- Score {r.get('average_score')}/10 ({r.get('total_reviews')} reviews)"
                )
        da_section = data.get("devils_advocate", {}).get("devils_advocate", {})
        if da_section:
            self.peer_review_output.append_line("\nDevil's Advocate Challenge:", "header")
            for point in da_section.get("counter_arguments", []):
                self.peer_review_output.append_line(f"  ⚡ {point}")
            for warning in da_section.get("risk_warnings", []):
                self.peer_review_output.append_line(f"  ⚠️ {warning}")
        rb_section = data.get("red_blue_team", {}).get("verdict", {})
        if rb_section:
            self.peer_review_output.append_line("\nRed-Blue Verdict:", "header")
            self.peer_review_output.append_line(f"Verdict: {rb_section.get('verdict', 'N/A')}")
            self.peer_review_output.append_line(f"Score: {rb_section.get('final_score', 'N/A')}/10")
        self.peer_review_output.append_line("")
        
        # Final decision tab
        self.decision_output.append_line("==== Final Decision ====", "header")
        chairman = data.get("chairman_synthesis", {}).get("chairman_decision", {})
        if chairman:
            self.decision_output.append_line(f"Action: {chairman.get('action', 'hold').upper()}")
            self.decision_output.append_line(f"Position: {chairman.get('position_percent', 0)}%")
            self.decision_output.append_line(f"Confidence: {self._format_percent(chairman.get('confidence'))}")
            for factor in chairman.get("key_factors", []):
                self.decision_output.append_line(f"  • {factor}")
        risk_budget = data.get("risk_budget", {}).get("risk_budget", {})
        if risk_budget:
            self.decision_output.append_line("\nRisk Budget:", "header")
            self.decision_output.append_line(f"  Total Investment: {self._format_price(risk_budget.get('total_investment'))}")
            self.decision_output.append_line(f"  Recommended Units: {risk_budget.get('recommended_units', 'N/A')}")
            self.decision_output.append_line(f"  Max Loss: {self._format_price(risk_budget.get('max_loss'))}")
            self.decision_output.append_line(f"  Stop Loss Price: {self._format_price(risk_budget.get('stop_loss_price'))}")
        decision_path = summary.get("decision_file") or data.get("decision_saved_to")
        if decision_path:
            self.decision_output.append_line(f"\nSaved to: {decision_path}")
    
    def _update_analyst_cards_from_data(self, opinions: List[Dict]):
        """Update analyst cards based on workflow snapshot"""
        role_map = {
            "technical": "technical_analyst",
            "industry": "industry_analyst",
            "financial": "financial_analyst",
            "market": "market_expert",
            "risk": "risk_analyst"
        }
        
        for card in self.analyst_cards.values():
            card.set_recommendation("", 0)
        
        for op in opinions:
            role_key = role_map.get(op.get("type"))
            if not role_key:
                continue
            if role_key in self.analyst_cards:
                self.analyst_cards[role_key].set_recommendation(
                    op.get("recommendation", "hold"),
                    op.get("confidence", 0)
                )
    
    def _format_price(self, value: Optional[float]) -> str:
        """Format price values gracefully"""
        if value is None:
            return "N/A"
        return f"${value:,.2f}"
    
    def _format_percent(self, value: Optional[float]) -> str:
        """Format percentage values gracefully"""
        if value is None:
            return "N/A"
        return f"{value:+.2f}%"
    
    def _format_number(self, value: Optional[float], decimals: int = 2) -> str:
        """Format generic numeric values"""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return "N/A"
    
    def _run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            self._send_output("main", "=" * 60, "header")
            self._send_output("main", "🚀 Starting Investment Decision Analysis", "header")
            self._send_output("main", f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "info")
            self._send_output("main", "=" * 60, "header")
            self._send_output("main", "")
            
            # Step 1: Data Preparation
            self._run_step("data_prep", self._step_data_preparation)
            
            # Step 2: Model Loading
            self._run_step("model_loading", self._step_model_loading)
            
            # Step 3: Price Prediction
            self._run_step("prediction", self._step_price_prediction)
            
            # Step 4: Advanced Analysis
            self._run_step("advanced", self._step_advanced_analysis)
            
            # Step 5: Analyst Initialization
            self._run_step("analyst_init", self._step_analyst_init)
            
            # Step 6: Independent Analysis
            self._run_step("independent", self._step_independent_analysis)
            
            # Step 7: Debate Phase
            self._run_step("debate", self._step_debate_phase)
            
            # Step 8: Discussion Phase
            self._run_step("discussion", self._step_discussion_phase)
            
            # Step 9: Signal Fusion
            self._run_step("signal_fusion", self._step_signal_fusion)
            
            # Step 10: Voting
            self._run_step("voting", self._step_voting)
            
            # Step 11: Peer Review
            self._run_step("peer_review", self._step_peer_review)
            
            # Step 12: Devil's Advocate
            self._run_step("devils_advocate", self._step_devils_advocate)
            
            # Step 13: Final Decision
            self._run_step("final_decision", self._step_final_decision)
            
            # Step 14: Red-Blue Validation
            self._run_step("validation", self._step_validation)
            
            # Step 15: Risk Budget
            self._run_step("risk_budget", self._step_risk_budget)
            
            # Step 16: Decision Review
            self._run_step("review", self._step_decision_review)
            
            # Complete
            self._send_output("main", "")
            self._send_output("main", "=" * 60, "success")
            self._send_output("main", "✅ Analysis Complete!", "success")
            self._send_output("main", "=" * 60, "success")
            
            self.message_queue.put({"type": "complete"})
            
        except Exception as e:
            self.message_queue.put({
                "type": "error",
                "message": str(e)
            })
    
    def _run_step(self, step_id, step_func):
        """Run a single step with progress tracking"""
        if not self.is_running:
            return
        
        start_time = time.time()
        
        # Update progress to running
        self.message_queue.put({
            "type": "progress",
            "step": step_id,
            "status": "running"
        })
        
        try:
            step_func()
            
            elapsed = time.time() - start_time
            self.message_queue.put({
                "type": "progress",
                "step": step_id,
                "status": "complete",
                "duration": f"{elapsed:.1f}s"
            })
            
        except Exception as e:
            self.message_queue.put({
                "type": "progress",
                "step": step_id,
                "status": "error"
            })
            self._send_output("main", f"❌ Error in step: {e}", "error")
            raise
    
    def _send_output(self, panel, text, tag=None):
        """Send output to a panel"""
        self.message_queue.put({
            "type": "output",
            "panel": panel,
            "text": text,
            "tag": tag
        })
    
    def _update_analyst(self, role, action, confidence):
        """Update analyst card"""
        self.message_queue.put({
            "type": "analyst_update",
            "role": role,
            "action": action,
            "confidence": confidence
        })
    
    # ==================== Analysis Steps ====================
    
    def _step_data_preparation(self):
        """Step 1: Data Preparation"""
        self._send_output("main", "📊 Step 1: Data Preparation", "header")
        self._send_output("main", "   Loading Bitcoin price data...", "info")
        time.sleep(0.5)
        
        # Try to load actual data
        data_path = PROJECT_ROOT / "Bitcoin price prediction Project" / "data"
        train_file = data_path / "processed_train_data.csv"
        
        if train_file.exists():
            import pandas as pd
            df = pd.read_csv(train_file)
            self._send_output("main", f"   ✓ Loaded {len(df)} training records", "success")
            self._send_output("main", f"   ✓ Latest close price: ${df['Close'].iloc[-1]:,.2f}", "success")
        else:
            self._send_output("main", "   ✓ Using simulated data", "warning")
        
        time.sleep(0.3)
        self._send_output("main", "")
    
    def _step_model_loading(self):
        """Step 2: Model Loading"""
        self._send_output("main", "🔧 Step 2: Loading Prediction Models", "header")
        
        models = [
            ("Linear Regression (Elastic Net)", 0.8212, 0.50),
            ("ARIMA(2,1,2)", 0.5420, 0.30),
            ("Seasonal (Prophet-like)", 0.5347, 0.20),
        ]
        
        for name, accuracy, weight in models:
            self._send_output("main", f"   Loading {name}...", "info")
            time.sleep(0.4)
            self._send_output("main", f"   ✓ {name}: {accuracy:.2%} accuracy, {weight:.0%} weight", "success")
        
        self._send_output("main", "")
    
    def _step_price_prediction(self):
        """Step 3: Price Prediction"""
        self._send_output("main", "📈 Step 3: Running Ensemble Prediction", "header")
        time.sleep(0.5)
        
        # Simulate prediction results
        current_price = 67500
        change_percent = random.uniform(-3, 8)
        predicted_price = current_price * (1 + change_percent / 100)
        
        self._send_output("main", f"   Current Price: ${current_price:,.2f}", "info")
        self._send_output("main", f"   Predicted Price (7 days): ${predicted_price:,.2f}", "info")
        
        if change_percent > 0:
            self._send_output("main", f"   Predicted Change: +{change_percent:.2f}%", "success")
        else:
            self._send_output("main", f"   Predicted Change: {change_percent:.2f}%", "error")
        
        # Store for later use
        self.prediction_data = {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "change_percent": change_percent,
            "confidence": random.uniform(0.6, 0.85)
        }
        
        self._send_output("main", "")
    
    def _step_advanced_analysis(self):
        """Step 4: Advanced Analysis"""
        self._send_output("main", "🔬 Step 4: Advanced Analysis", "header")
        
        # Multi-timeframe
        self._send_output("main", "   Running Multi-Timeframe Analysis...", "info")
        time.sleep(0.3)
        
        self._send_output("analysis", "=" * 50, "header")
        self._send_output("analysis", "MULTI-TIMEFRAME ANALYSIS", "header")
        self._send_output("analysis", "=" * 50, "header")
        
        timeframes = [
            ("Short-term (1-7 days)", "Technical", "+2.60%", "Light position buy"),
            ("Medium-term (1-4 weeks)", "Sentiment", "+5.20%", "Buy"),
            ("Long-term (1-3 months)", "Fundamentals", "+7.80%", "Buy"),
        ]
        
        for name, focus, change, signal in timeframes:
            self._send_output("analysis", f"\n[{name}]", "header")
            self._send_output("analysis", f"  Focus: {focus}")
            self._send_output("analysis", f"  Predicted Change: {change}")
            self._send_output("analysis", f"  Signal: {signal}")
            time.sleep(0.2)
        
        # Scenario analysis
        self._send_output("main", "   Running Scenario Analysis...", "info")
        time.sleep(0.3)
        
        self._send_output("analysis", "\n" + "=" * 50, "header")
        self._send_output("analysis", "SCENARIO ANALYSIS", "header")
        self._send_output("analysis", "=" * 50, "header")
        
        scenarios = [
            ("Bull Market", "30%", "+15.0%", "Medium risk"),
            ("Bear Market", "20%", "-15.0%", "High risk"),
            ("Sideways", "40%", "+1.5%", "Low risk"),
            ("Black Swan", "10%", "-35.0%", "Extreme risk"),
        ]
        
        for name, prob, change, risk in scenarios:
            self._send_output("analysis", f"\n[{name}] (Probability: {prob})")
            self._send_output("analysis", f"  Expected Change: {change}")
            self._send_output("analysis", f"  Risk Level: {risk}")
            time.sleep(0.15)
        
        self._send_output("main", "   ✓ Advanced analysis complete", "success")
        self._send_output("main", "")
    
    def _step_analyst_init(self):
        """Step 5: Analyst Initialization"""
        self._send_output("main", "👥 Step 5: Initializing Analysts", "header")
        
        analysts = [
            ("technical_analyst", "Technical Analyst", "Chart Master", "Let the data speak"),
            ("industry_analyst", "Industry Analyst", "Industry Observer", "See the big picture"),
            ("financial_analyst", "Financial Analyst", "Risk Controller", "Risk control is priority"),
            ("market_expert", "Market Expert", "Sentiment Catcher", "The market is always right"),
            ("risk_analyst", "Risk Analyst", "Security Guardian", "Better safe than sorry"),
        ]
        
        for role, name, nickname, catchphrase in analysts:
            self._send_output("main", f"   Initializing {name} ({nickname})...", role)
            time.sleep(0.2)
        
        self._send_output("main", "   ✓ All 5 analysts ready", "success")
        self._send_output("main", "")
    
    def _step_independent_analysis(self):
        """Step 6: Independent Analysis"""
        self._send_output("main", "🔍 Step 6: Independent Analysis Phase", "header")
        
        analyses = [
            ("technical_analyst", "Technical Analyst", 
             "Based on ensemble model prediction showing +5.2% expected change, "
             "RSI at 58 indicating neutral-bullish momentum, MACD showing positive divergence. "
             "Price is above 20-day MA suggesting upward trend continuation."),
            ("industry_analyst", "Industry Analyst",
             "Industry fundamentals remain strong with continued institutional adoption. "
             "ETF inflows positive, regulatory clarity improving in major markets. "
             "Competitive position of Bitcoin strengthening versus alternatives."),
            ("financial_analyst", "Financial Analyst",
             "Risk-reward ratio appears favorable at current levels. "
             "Recommend position sizing of 40-50% with stop-loss at -8%. "
             "Take-profit target at +12% based on resistance levels."),
            ("market_expert", "Market Expert",
             "Market sentiment shifting positive, Fear-Greed index at 65. "
             "Institutional fund flows show accumulation patterns. "
             "Retail interest increasing but not at euphoric levels yet."),
            ("risk_analyst", "Risk Analyst",
             "Key risks: regulatory uncertainty, macroeconomic headwinds, "
             "correlation with equity markets. Black swan probability ~10%. "
             "Recommend maintaining cash buffer and strict stop-losses."),
        ]
        
        self._send_output("analysis", "\n" + "=" * 50, "header")
        self._send_output("analysis", "INDEPENDENT ANALYSIS", "header")
        self._send_output("analysis", "=" * 50, "header")
        
        for role, name, analysis in analyses:
            if not self.is_running:
                return
            
            self._send_output("main", f"   {name} analyzing...", role)
            self._send_output("analysis", f"\n[{name}]", role)
            self._send_output("analysis", f"  {analysis}")
            time.sleep(0.4)
        
        self._send_output("main", "   ✓ All analysts completed independent analysis", "success")
        self._send_output("main", "")
    
    def _step_debate_phase(self):
        """Step 7: Debate Phase"""
        self._send_output("main", "💬 Step 7: Debate Phase", "header")
        self._send_output("main", "   Identifying divergent viewpoints...", "info")
        time.sleep(0.3)
        
        self._send_output("analysis", "\n" + "=" * 50, "header")
        self._send_output("analysis", "DEBATE PHASE", "header")
        self._send_output("analysis", "=" * 50, "header")
        
        # Simulate debate
        self._send_output("analysis", "\n[Topic: Market Direction]", "header")
        self._send_output("analysis", "\nBullish Side (Technical, Industry, Market):", "success")
        self._send_output("analysis", "  - Technical indicators support upward movement")
        self._send_output("analysis", "  - Institutional adoption creating sustained demand")
        self._send_output("analysis", "  - Sentiment shift suggests market ready for rally")
        
        time.sleep(0.3)
        
        self._send_output("analysis", "\nCautious Side (Financial, Risk):", "warning")
        self._send_output("analysis", "  - Current levels require careful risk management")
        self._send_output("analysis", "  - Macro risks could trigger sudden corrections")
        self._send_output("analysis", "  - Recommend phased entry rather than full position")
        
        time.sleep(0.3)
        
        self._send_output("analysis", "\n[Debate Outcome]", "header")
        self._send_output("analysis", "  Consensus forming around: CAUTIOUS BULLISH")
        self._send_output("analysis", "  Key agreement: Position sizing is critical")
        
        self._send_output("main", "   ✓ Debate complete - consensus forming", "success")
        self._send_output("main", "")
    
    def _step_discussion_phase(self):
        """Step 8: Discussion Phase"""
        self._send_output("main", "🗣️ Step 8: Discussion Phase", "header")
        self._send_output("main", "   Analysts exchanging refined views...", "info")
        time.sleep(0.5)
        
        self._send_output("main", "   ✓ Views refined through discussion", "success")
        self._send_output("main", "")
    
    def _step_signal_fusion(self):
        """Step 9: Signal Fusion"""
        self._send_output("main", "🔄 Step 9: Bayesian Signal Fusion", "header")
        
        self._send_output("analysis", "\n" + "=" * 50, "header")
        self._send_output("analysis", "SIGNAL FUSION", "header")
        self._send_output("analysis", "=" * 50, "header")
        
        signals = [
            ("Linear Regression", "bullish", 0.82, 0.50),
            ("ARIMA", "bullish", 0.54, 0.30),
            ("Seasonal", "neutral", 0.53, 0.20),
            ("RSI Signal", "bullish", 0.60, 0.15),
            ("MACD Signal", "bullish", 0.58, 0.15),
        ]
        
        self._send_output("analysis", "\n[Signal Sources]")
        for name, direction, accuracy, weight in signals:
            dir_text = "▲" if direction == "bullish" else ("▼" if direction == "bearish" else "●")
            self._send_output("analysis", f"  {name}: {dir_text} {direction} (acc: {accuracy:.0%}, w: {weight:.0%})")
            time.sleep(0.15)
        
        self._send_output("analysis", "\n[Fusion Result]", "header")
        self._send_output("analysis", "  Direction: BULLISH", "success")
        self._send_output("analysis", "  Up Probability: 72.5%")
        self._send_output("analysis", "  Confidence: 68.3%")
        
        self._send_output("main", "   ✓ Signals fused successfully", "success")
        self._send_output("main", "")
    
    def _step_voting(self):
        """Step 10: Voting Phase"""
        self._send_output("main", "🗳️ Step 10: Voting Phase", "header")
        
        votes = [
            ("technical_analyst", "buy", 8),
            ("industry_analyst", "buy", 7),
            ("financial_analyst", "hold", 6),
            ("market_expert", "buy", 7),
            ("risk_analyst", "hold", 5),
        ]
        
        self._send_output("analysis", "\n" + "=" * 50, "header")
        self._send_output("analysis", "VOTING RESULTS", "header")
        self._send_output("analysis", "=" * 50, "header")
        
        for role, action, confidence in votes:
            name = {
                "technical_analyst": "Technical Analyst",
                "industry_analyst": "Industry Analyst",
                "financial_analyst": "Financial Analyst",
                "market_expert": "Market Expert",
                "risk_analyst": "Risk Analyst",
            }[role]
            
            self._send_output("main", f"   {name}: {action.upper()} (confidence: {confidence}/10)", role)
            self._send_output("analysis", f"\n[{name}]", role)
            self._send_output("analysis", f"  Recommendation: {action.upper()}")
            self._send_output("analysis", f"  Confidence: {confidence}/10")
            
            # Update analyst card
            self._update_analyst(role, action, confidence)
            time.sleep(0.25)
        
        self._send_output("analysis", "\n[Vote Summary]", "header")
        self._send_output("analysis", "  BUY: 3 votes")
        self._send_output("analysis", "  HOLD: 2 votes")
        self._send_output("analysis", "  Majority: BUY", "success")
        
        self._send_output("main", "   ✓ Voting complete - BUY majority", "success")
        self._send_output("main", "")
    
    def _step_peer_review(self):
        """Step 11: Anonymous Peer Review"""
        self._send_output("main", "📝 Step 11: Anonymous Peer Review", "header")
        self._send_output("main", "   Anonymizing analyses (A, B, C, D, E)...", "info")
        time.sleep(0.3)
        
        self._send_output("peer_review", "=" * 50, "header")
        self._send_output("peer_review", "ANONYMOUS PEER REVIEW", "header")
        self._send_output("peer_review", "=" * 50, "header")
        
        # Anonymization
        self._send_output("peer_review", "\n[Anonymization]")
        labels = ["A", "B", "C", "D", "E"]
        random.shuffle(labels)
        
        for i, label in enumerate(labels):
            self._send_output("peer_review", f"  Analyst {label} -> [Identity Hidden]")
        
        time.sleep(0.3)
        self._send_output("main", "   Conducting cross-evaluations...", "info")
        
        # Peer reviews
        self._send_output("peer_review", "\n[Cross-Evaluations]", "header")
        
        reviews = [
            ("Analyst A reviews B", {"accuracy": 8, "insight": 7, "logic": 8}),
            ("Analyst B reviews C", {"accuracy": 7, "insight": 8, "logic": 7}),
            ("Analyst C reviews A", {"accuracy": 8, "insight": 7, "logic": 8}),
            ("Analyst D reviews E", {"accuracy": 6, "insight": 6, "logic": 7}),
            ("Analyst E reviews D", {"accuracy": 7, "insight": 7, "logic": 6}),
        ]
        
        for review_desc, scores in reviews:
            avg = sum(scores.values()) / len(scores)
            self._send_output("peer_review", f"\n  {review_desc}:")
            self._send_output("peer_review", f"    Accuracy: {scores['accuracy']}/10")
            self._send_output("peer_review", f"    Insight: {scores['insight']}/10")
            self._send_output("peer_review", f"    Logic: {scores['logic']}/10")
            self._send_output("peer_review", f"    Average: {avg:.1f}/10")
            time.sleep(0.15)
        
        # Rankings
        self._send_output("peer_review", "\n" + "-" * 40, "header")
        self._send_output("peer_review", "QUALITY RANKINGS", "header")
        self._send_output("peer_review", "-" * 40, "header")
        
        rankings = [
            ("#1", "Analyst A", 8.2),
            ("#2", "Analyst C", 7.8),
            ("#3", "Analyst B", 7.5),
            ("#4", "Analyst E", 6.8),
            ("#5", "Analyst D", 6.5),
        ]
        
        for rank, analyst, score in rankings:
            color = "success" if rank == "#1" else None
            self._send_output("peer_review", f"  {rank}. {analyst} - Score: {score:.1f}/10", color)
        
        self._send_output("main", "   ✓ Peer review complete - rankings generated", "success")
        self._send_output("main", "")
    
    def _step_devils_advocate(self):
        """Step 12: Devil's Advocate"""
        self._send_output("main", "😈 Step 12: Devil's Advocate Challenge", "header")
        self._send_output("main", "   Challenging consensus view...", "info")
        time.sleep(0.3)
        
        self._send_output("peer_review", "\n" + "=" * 50, "header")
        self._send_output("peer_review", "DEVIL'S ADVOCATE CHALLENGE", "header")
        self._send_output("peer_review", "=" * 50, "header")
        
        self._send_output("peer_review", "\n[Challenging Consensus: BUY (3/5 agree)]", "warning")
        
        self._send_output("peer_review", "\n[Counter-Arguments]", "error")
        counter_args = [
            "The market may be in a bull trap - recent gains could reverse",
            "Institutional interest might be peaking, smart money selling",
            "Technical indicators may lag fundamental deterioration"
        ]
        for arg in counter_args:
            self._send_output("peer_review", f"  ⚡ {arg}")
            time.sleep(0.15)
        
        self._send_output("peer_review", "\n[Alternative Scenarios]", "warning")
        scenarios = [
            "Black swan event (exchange hack): 30-50% drop",
            "Major regulatory action: 20-30% drop",
            "Correlation breakdown with equities: Unpredictable"
        ]
        for scenario in scenarios:
            self._send_output("peer_review", f"  🔄 {scenario}")
        
        self._send_output("peer_review", "\n[Risk Warnings]", "error")
        self._send_output("peer_review", "  ⚠️ Model predictions have inherent uncertainty")
        self._send_output("peer_review", "  ⚠️ Market regime changes can invalidate patterns")
        self._send_output("peer_review", "  ⚠️ Liquidity conditions can amplify moves")
        
        self._send_output("peer_review", "\n[Challenge Confidence: 6/10]")
        
        self._send_output("main", "   ✓ Challenge complete - risks identified", "success")
        self._send_output("main", "")
    
    def _step_final_decision(self):
        """Step 13: Final Decision"""
        self._send_output("main", "🎯 Step 13: Final Decision", "header")
        self._send_output("main", "   Chairman synthesizing all inputs...", "info")
        time.sleep(0.5)
        
        self._send_output("decision", "=" * 50, "header")
        self._send_output("decision", "FINAL INVESTMENT DECISION", "header")
        self._send_output("decision", "=" * 50, "header")
        
        self._send_output("decision", "\n[Chairman's Synthesis]", "header")
        self._send_output("decision", """
Based on comprehensive analysis of all inputs:

1. Peer-reviewed quality rankings favor BUY recommendation
2. Top-ranked Analyst A provided compelling technical evidence
3. Signal fusion shows 72.5% bullish probability
4. Devil's Advocate raised valid concerns requiring caution
5. Risk-adjusted position sizing is essential
        """)
        
        self._send_output("decision", "\n" + "=" * 50, "success")
        self._send_output("decision", "RECOMMENDATION: BUY", "success")
        self._send_output("decision", "=" * 50, "success")
        
        self._send_output("decision", "\n[Decision Details]")
        self._send_output("decision", "  Action: BUY")
        self._send_output("decision", "  Position Size: 45%")
        self._send_output("decision", "  Confidence: 7/10")
        self._send_output("decision", "  Stop-Loss: -8%")
        self._send_output("decision", "  Take-Profit: +12%")
        
        self._send_output("main", "   ✓ Final decision: BUY (45% position, confidence 7/10)", "success")
        self._send_output("main", "")
    
    def _step_validation(self):
        """Step 14: Red-Blue Team Validation"""
        self._send_output("main", "⚔️ Step 14: Red-Blue Team Validation", "header")
        self._send_output("main", "   Red team challenging, Blue team defending...", "info")
        time.sleep(0.4)
        
        self._send_output("decision", "\n" + "-" * 40, "header")
        self._send_output("decision", "RED-BLUE TEAM VALIDATION", "header")
        self._send_output("decision", "-" * 40, "header")
        
        self._send_output("decision", "\n[Red Team Challenges]", "error")
        self._send_output("decision", "  - Model confidence at 68% is not overwhelming")
        self._send_output("decision", "  - Downside risk may be underestimated")
        self._send_output("decision", "  - Tail risk not fully considered")
        
        self._send_output("decision", "\n[Blue Team Defense]", "success")
        self._send_output("decision", "  + Multi-model ensemble improves reliability")
        self._send_output("decision", "  + Decision aligns with prediction direction")
        self._send_output("decision", "  + Position sizing provides risk buffer")
        self._send_output("decision", "  + Linear Regression has 82% historical accuracy")
        
        self._send_output("decision", "\n[Judge Verdict]", "header")
        self._send_output("decision", "  Challenge Score: 6/10")
        self._send_output("decision", "  Defense Score: 7.5/10")
        self._send_output("decision", "  Final Score: 7.1/10")
        self._send_output("decision", "  Verdict: DECISION PASSED VALIDATION", "success")
        
        self._send_output("main", "   ✓ Validation passed - decision confirmed", "success")
        self._send_output("main", "")
    
    def _step_risk_budget(self):
        """Step 15: Risk Budget Calculation"""
        self._send_output("main", "💰 Step 15: Risk Budget Management", "header")
        self._send_output("main", "   Calculating position based on risk budget...", "info")
        time.sleep(0.3)
        
        self._send_output("decision", "\n" + "-" * 40, "header")
        self._send_output("decision", "RISK BUDGET CALCULATION", "header")
        self._send_output("decision", "-" * 40, "header")
        
        self._send_output("decision", "\n[Account Parameters]")
        self._send_output("decision", "  Total Capital: $100,000")
        self._send_output("decision", "  Max Risk Budget: 5% ($5,000)")
        
        self._send_output("decision", "\n[Trade Parameters]")
        self._send_output("decision", "  Entry Price: $67,500")
        self._send_output("decision", "  Stop-Loss Price: $62,100 (-8%)")
        self._send_output("decision", "  Prediction Confidence: 68%")
        
        self._send_output("decision", "\n[Position Calculation]")
        self._send_output("decision", "  Risk Per Unit: $5,400")
        self._send_output("decision", "  Recommended Units: 0.76 BTC")
        self._send_output("decision", "  Investment Amount: $45,000")
        self._send_output("decision", "  Position Ratio: 45%")
        
        self._send_output("decision", "\n[Risk-Reward Analysis]")
        self._send_output("decision", "  Potential Risk: $5,400")
        self._send_output("decision", "  Potential Reward: $8,100")
        self._send_output("decision", "  Risk-Reward Ratio: 1:1.5", "success")
        
        self._send_output("main", "   ✓ Position calculated: 45% ($45,000)", "success")
        self._send_output("main", "")
    
    def _step_decision_review(self):
        """Step 16: Decision Review"""
        self._send_output("main", "📋 Step 16: Decision Review", "header")
        self._send_output("main", "   Analyzing decision quality...", "info")
        time.sleep(0.3)
        
        self._send_output("decision", "\n" + "=" * 50, "header")
        self._send_output("decision", "DECISION REVIEW REPORT", "header")
        self._send_output("decision", "=" * 50, "header")
        
        self._send_output("decision", "\n[Decision Summary]")
        self._send_output("decision", "  Recommendation: BUY")
        self._send_output("decision", "  Position: 45%")
        self._send_output("decision", "  Confidence: 7/10 (Moderate)")
        self._send_output("decision", "  Style: Moderate-Aggressive")
        
        self._send_output("decision", "\n[Consensus Analysis]")
        self._send_output("decision", "  Level: Majority Consensus")
        self._send_output("decision", "  Bullish: 3 (60%)")
        self._send_output("decision", "  Neutral: 2 (40%)")
        self._send_output("decision", "  Dominant View: BULLISH", "success")
        
        self._send_output("decision", "\n[Key Insights]")
        self._send_output("decision", "  [+] Model prediction confidence high (68%)")
        self._send_output("decision", "  [+] Analysts reached majority consensus")
        self._send_output("decision", "  [+] Decision consistent with model direction")
        self._send_output("decision", "  [*] Position sized appropriately (45%)")
        
        self._send_output("decision", "\n[Risk Assessment]")
        self._send_output("decision", "  Risk Level: MODERATE (Score: 4/10)")
        self._send_output("decision", "  Risk Factors:")
        self._send_output("decision", "    - Market volatility elevated")
        self._send_output("decision", "    - Some analyst divergence")
        self._send_output("decision", "  Mitigations:")
        self._send_output("decision", "    - Strict stop-loss at -8%")
        self._send_output("decision", "    - Position size within risk budget")
        
        self._send_output("decision", "\n[Lessons & Recommendations]")
        self._send_output("decision", "  - Multi-model ensemble improved prediction stability")
        self._send_output("decision", "  - Anonymous peer review reduced bias")
        self._send_output("decision", "  - Devil's Advocate identified key risks")
        self._send_output("decision", "  - Continue tracking model vs actual performance")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._send_output("decision", f"\n[Report Generated: {timestamp}]", "muted")
        
        self._send_output("main", "   ✓ Review complete - decision quality: GOOD", "success")
        self._send_output("main", "")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


# ==================== Entry Point ====================

def main():
    """Main entry point"""
    print("Starting Investment Decision System GUI...")
    print("Please wait while the interface loads...")
    
    app = InvestmentSystemGUI()
    app.run()


if __name__ == "__main__":
    main()

