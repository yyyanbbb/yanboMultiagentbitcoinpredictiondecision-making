# -*- coding: utf-8 -*-
"""
Output Formatter Utility
Provides elegant terminal output formatting
"""

from colorama import Fore, Style, init
from datetime import datetime
from typing import List, Dict, Optional

# Initialize colorama
init(autoreset=True)


class OutputFormatter:
    """Output formatter utility class"""
    
    # Color theme
    THEME = {
        "primary": Fore.CYAN,
        "secondary": Fore.YELLOW,
        "success": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED,
        "info": Fore.WHITE,
        "muted": Fore.LIGHTBLACK_EX,
        "bullish": Fore.GREEN,
        "bearish": Fore.RED,
        "neutral": Fore.YELLOW
    }
    
    # Icons
    ICONS = {
        "chart": "[Chart]",
        "industry": "[Industry]",
        "finance": "[Finance]",
        "market": "[Market]",
        "risk": "[Risk]",
        "manager": "[Manager]",
        "ok": "[OK]",
        "error": "[X]",
        "warning": "[!]",
        "info": "[*]",
        "arrow_up": "[UP]",
        "arrow_down": "[DOWN]",
        "arrow_right": "->",
        "bullet": "-",
        "star": "*",
        "check": "[v]",
        "cross": "[x]"
    }
    
    @classmethod
    def print_banner(cls, title: str, subtitle: str = "", width: int = 78):
        """Print banner"""
        border = "=" * width
        print(f"\n{cls.THEME['primary']}+{border}+")
        print(f"|{title:^{width}}|")
        if subtitle:
            print(f"|{subtitle:^{width}}|")
        print(f"+{border}+{Style.RESET_ALL}\n")
    
    @classmethod
    def print_section(cls, title: str, icon: str = "", color: str = "primary"):
        """Print section title"""
        theme_color = cls.THEME.get(color, cls.THEME["primary"])
        separator = "=" * 78
        icon_str = cls.ICONS.get(icon, icon) if icon else ""
        full_title = f"{icon_str} {title}" if icon_str else title
        print(f"\n{theme_color}+{separator}+")
        print(f"|{full_title:^78}|")
        print(f"+{separator}+{Style.RESET_ALL}\n")
    
    @classmethod
    def print_subsection(cls, title: str, color: str = "secondary"):
        """Print subsection title"""
        theme_color = cls.THEME.get(color, cls.THEME["secondary"])
        print(f"\n{theme_color}--- {title} ---{Style.RESET_ALL}")
    
    @classmethod
    def print_info(cls, message: str, color: str = "info", indent: int = 0):
        """Print info message"""
        theme_color = cls.THEME.get(color, cls.THEME["info"])
        prefix = "  " * indent
        print(f"{prefix}{theme_color}{message}{Style.RESET_ALL}")
    
    @classmethod
    def print_success(cls, message: str, indent: int = 0):
        """Print success message"""
        cls.print_info(f"{cls.ICONS['ok']} {message}", "success", indent)
    
    @classmethod
    def print_error(cls, message: str, indent: int = 0):
        """Print error message"""
        cls.print_info(f"{cls.ICONS['error']} {message}", "error", indent)
    
    @classmethod
    def print_warning(cls, message: str, indent: int = 0):
        """Print warning message"""
        cls.print_info(f"{cls.ICONS['warning']} {message}", "warning", indent)
    
    @classmethod
    def print_bullet(cls, message: str, indent: int = 1):
        """Print bullet item"""
        cls.print_info(f"{cls.ICONS['bullet']} {message}", "info", indent)
    
    @classmethod
    def print_analyst(cls, role: str, name: str, message: str):
        """Print analyst speech"""
        icon = cls.ICONS.get(role, "[A]")
        print(f"\n{cls.THEME['primary']}{icon} {name}{Style.RESET_ALL}")
        print(f"  {cls.THEME['info']}{message}{Style.RESET_ALL}")
    
    @classmethod
    def print_recommendation(cls, action: str, confidence: int, reason: str = ""):
        """Print investment recommendation"""
        # Select color based on recommendation type
        if action in ["strong_buy", "buy"]:
            color = cls.THEME["bullish"]
            icon = cls.ICONS["arrow_up"]
        elif action in ["strong_sell", "sell"]:
            color = cls.THEME["bearish"]
            icon = cls.ICONS["arrow_down"]
        else:
            color = cls.THEME["neutral"]
            icon = cls.ICONS["arrow_right"]
        
        action_display = {
            "strong_buy": "STRONG BUY",
            "buy": "BUY",
            "hold": "HOLD",
            "sell": "SELL",
            "strong_sell": "STRONG SELL"
        }.get(action, action)
        
        print(f"  {color}{icon} {action_display}{Style.RESET_ALL} | Confidence: {confidence}/10")
        if reason:
            print(f"    {cls.THEME['muted']}{reason[:80]}...{Style.RESET_ALL}")
    
    @classmethod
    def print_price(cls, label: str, price: float, change: float = None):
        """Print price info"""
        price_str = f"${price:,.2f}"
        print(f"  {cls.THEME['info']}{label}: {price_str}", end="")
        
        if change is not None:
            if change > 0:
                print(f" {cls.THEME['bullish']}(+{change:.2f}%){Style.RESET_ALL}")
            elif change < 0:
                print(f" {cls.THEME['bearish']}({change:.2f}%){Style.RESET_ALL}")
            else:
                print(f" {cls.THEME['neutral']}(0.00%){Style.RESET_ALL}")
        else:
            print(Style.RESET_ALL)
    
    @classmethod
    def print_table(cls, headers: list, rows: list, widths: list = None):
        """Print table"""
        if not widths:
            widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2 
                     for i, h in enumerate(headers)]
        
        # Table header
        border = "+" + "+".join("-" * w for w in widths) + "+"
        header_row = "|" + "|".join(f"{str(h):^{w}}" for h, w in zip(headers, widths)) + "|"
        
        print(f"{cls.THEME['primary']}{border}")
        print(f"{header_row}")
        print(f"{border}{Style.RESET_ALL}")
        
        # Data rows
        for row in rows:
            row_str = "|" + "|".join(f"{str(c):^{w}}" for c, w in zip(row, widths)) + "|"
            print(f"{cls.THEME['info']}{row_str}{Style.RESET_ALL}")
        
        print(f"{cls.THEME['primary']}{border}{Style.RESET_ALL}")
    
    @classmethod
    def print_progress(cls, current: int, total: int, label: str = ""):
        """Print progress bar"""
        width = 40
        progress = int(width * current / total)
        bar = "#" * progress + "-" * (width - progress)
        percent = current / total * 100
        
        print(f"\r  {label} [{bar}] {percent:.1f}%", end="", flush=True)
        if current >= total:
            print()  # Newline after completion
    
    @classmethod
    def print_divider(cls, char: str = "-", width: int = 78):
        """Print divider line"""
        print(f"{cls.THEME['muted']}{char * width}{Style.RESET_ALL}")
    
    @classmethod
    def print_timestamp(cls, label: str = "Time"):
        """Print timestamp"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{cls.THEME['muted']}{label}: {now}{Style.RESET_ALL}")
    
    @classmethod
    def format_trend(cls, trend: str) -> str:
        """Format trend display"""
        trend_map = {
            "strong_upward": f"{cls.THEME['bullish']}STRONG UPWARD [UP][UP]{Style.RESET_ALL}",
            "upward": f"{cls.THEME['bullish']}RISING [UP]{Style.RESET_ALL}",
            "sideways": f"{cls.THEME['neutral']}SIDEWAYS [-]{Style.RESET_ALL}",
            "downward": f"{cls.THEME['bearish']}DECLINING [DOWN]{Style.RESET_ALL}",
            "strong_downward": f"{cls.THEME['bearish']}STRONG DOWNWARD [DOWN][DOWN]{Style.RESET_ALL}"
        }
        return trend_map.get(trend, trend)
    
    @classmethod
    def format_confidence(cls, confidence: float) -> str:
        """Format confidence display"""
        if confidence >= 0.8:
            return f"{cls.THEME['success']}{confidence:.1%} (HIGH){Style.RESET_ALL}"
        elif confidence >= 0.6:
            return f"{cls.THEME['warning']}{confidence:.1%} (MEDIUM){Style.RESET_ALL}"
        else:
            return f"{cls.THEME['error']}{confidence:.1%} (LOW){Style.RESET_ALL}"
    
    @classmethod
    def print_decision_box(cls, action: str, position: int, confidence: int):
        """Print decision box"""
        action_display = {
            "strong_buy": "STRONG BUY",
            "buy": "BUY",
            "hold": "HOLD",
            "sell": "SELL",
            "strong_sell": "STRONG SELL"
        }.get(action, action)
        
        # Select color
        if action in ["strong_buy", "buy"]:
            color = cls.THEME["bullish"]
        elif action in ["strong_sell", "sell"]:
            color = cls.THEME["bearish"]
        else:
            color = cls.THEME["neutral"]
        
        box_width = 50
        border = "=" * box_width
        
        print(f"\n{color}+{border}+")
        print(f"|{'FINAL INVESTMENT DECISION':^{box_width}}|")
        print(f"+{border}+")
        print(f"|  Recommendation: {action_display:<{box_width-20}}|")
        print(f"|  Position: {position}%{' ' * (box_width-15)}|")
        print(f"|  Confidence: {confidence}/10{' ' * (box_width-17)}|")
        print(f"+{border}+{Style.RESET_ALL}\n")


class ProgressTracker:
    """Progress tracker utility"""
    
    def __init__(self, total_steps: int, description: str = ""):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: int = None, message: str = ""):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        OutputFormatter.print_progress(
            self.current_step, 
            self.total_steps, 
            f"{self.description}: {message}" if message else self.description
        )
    
    def complete(self):
        """Complete progress"""
        self.current_step = self.total_steps
        elapsed = (datetime.now() - self.start_time).total_seconds()
        OutputFormatter.print_success(f"{self.description} completed (elapsed: {elapsed:.1f}s)")


# Test
if __name__ == "__main__":
    print("Output Formatter Utility Test\n")
    
    # Test banner
    OutputFormatter.print_banner("Intelligent Investment Decision System", "v3.0 Enhanced Version")
    
    # Test section
    OutputFormatter.print_section("Data Preparation Phase", "chart")
    
    # Test info output
    OutputFormatter.print_success("Configuration validation passed")
    OutputFormatter.print_warning("Model is loading")
    OutputFormatter.print_error("Connection failed")
    
    # Test price
    OutputFormatter.print_price("Current Price", 67500, 5.23)
    OutputFormatter.print_price("Predicted Price", 71000, -2.15)
    
    # Test recommendation
    OutputFormatter.print_subsection("Investment Recommendation")
    OutputFormatter.print_recommendation("buy", 7, "Based on technical analysis and market sentiment")
    OutputFormatter.print_recommendation("sell", 5, "High risk, recommend reducing position")
    
    # Test table
    OutputFormatter.print_subsection("Model Comparison")
    headers = ["Model", "Accuracy", "Weight"]
    rows = [
        ["Linear Regression", "82.1%", "50%"],
        ["ARIMA", "54.2%", "30%"],
        ["Prophet", "53.5%", "20%"]
    ]
    OutputFormatter.print_table(headers, rows, [20, 12, 10])
    
    # Test decision box
    OutputFormatter.print_decision_box("buy", 60, 7)
    
    # Test progress
    import time
    tracker = ProgressTracker(5, "Analysis")
    for i in range(5):
        time.sleep(0.3)
        tracker.update(message=f"Step {i+1}")
    tracker.complete()
