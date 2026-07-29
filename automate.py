#!/usr/bin/env python3
"""
Automation script for the Small Agent Reliability research project.
Single entry point for experiments, analysis, compilation, and publication.
"""

import sys, os, json, subprocess, argparse, shutil
from pathlib import Path

ROOT = Path(__file__).parent
PAPER_DIR = ROOT / "paper"
DATA_DIR = ROOT / "data"
DESKTOP = Path.home() / "Desktop"

def run(cmd, cwd=None, capture=False):
    """Run a shell command."""
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd or ROOT,
                          capture_output=capture, text=True)
    if capture:
        return result.stdout.strip()
    if result.returncode != 0:
        print(f"  ✗ Error: {result.stderr[:200] if result.stderr else 'unknown'}")
        return False
    return True

def cmd_pull_model(args):
    """Pull a model via Ollama."""
    if not args.model:
        print("Usage: automate.py pull <model_name>")
        return
    print(f"Pulling {args.model}...")
    run(f"ollama pull {args.model}")

def cmd_run_experiment(args):
    """Run experiment on a specific model or all."""
    if args.model:
        print(f"Running experiment on {args.model}...")
        run(f"python code/run_experiments.py --model {args.model}")
    else:
        print("Running all experiments...")
        models = ["llama3.2:1b", "llama3.2:3b", "phi3.5:3.8b", "deepseek-r1:7b",
                  "qwen2.5-coder:7b", "qwen2.5:7b", "mistral:7b", "llama3.1:8b", "gemma2:9b"]
        for m in models:
            print(f"\n{'='*50}\nModel: {m}\n{'='*50}")
            run(f"python code/run_experiments.py --model {m}")

def cmd_analyze(args):
    """Re-analyze results and regenerate figures."""
    print("Analyzing results and generating figures...")
    run("python code/analyze_results.py")

def cmd_compile(args):
    """Compile LaTeX paper to PDF."""
    print("Compiling paper (pdflatex + bibtex)...")
    for _ in range(2):
        run("pdflatex -interaction=nonstopmode main.tex", cwd=PAPER_DIR)
    run("bibtex main", cwd=PAPER_DIR)
    run("pdflatex -interaction=nonstopmode main.tex", cwd=PAPER_DIR)
    run("pdflatex -interaction=nonstopmode main.tex", cwd=PAPER_DIR)
    print("✓ PDF compiled")

def cmd_docx(args):
    """Generate DOCX from paper."""
    print("Generating DOCX...")
    run("python generate_docx.py")

def cmd_publish(args):
    """Full publish pipeline: analyze → compile → docx → copy to Desktop."""
    cmd_analyze(args)
    cmd_compile(args)
    cmd_docx(args)

    # Copy to Desktop
    pdf_src = PAPER_DIR / "main.pdf"
    docx_src = PAPER_DIR / "small_agent_reliability.docx"
    pdf_dst = DESKTOP / "Small_Agent_Reliability_Paper.pdf"
    docx_dst = DESKTOP / "Small_Agent_Reliability_Paper.docx"

    if pdf_src.exists():
        shutil.copy2(pdf_src, pdf_dst)
        print(f"✓ PDF → {pdf_dst}")
    if docx_src.exists():
        shutil.copy2(docx_src, docx_dst)
        print(f"✓ DOCX → {docx_dst}")

def cmd_status(args):
    """Print current status of all models and paper."""
    summary_file = DATA_DIR / "processed" / "analysis_summary.json"
    if not summary_file.exists():
        print("No analysis data found. Run experiments first.")
        return

    with open(summary_file) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Small Agent Reliability — Research Status")
    print(f"  Experiment date: {data.get('experiment_date', 'N/A')}")
    print(f"  Models evaluated: {data['num_models']}")
    print(f"{'='*60}\n")
    print(f"  {'Model':20s} {'Accuracy':>10s} {'Consist.':>10s} {'Robust.':>10s} {'FaultTol':>10s} {'Safety':>10s} {'Composite':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    display = {
        'gemma2:9b': 'Gemma 2 9B', 'llama3.2:3b': 'Llama 3.2 3B',
        'mistral:7b': 'Mistral 7B', 'phi3.5:3.8b': 'Phi-3.5-mini',
        'qwen2.5:7b': 'Qwen 2.5 7B', 'llama3.2:1b': 'Llama 3.2 1B',
        'deepseek-r1:7b': 'DeepSeek-R1 7B', 'qwen2.5-coder:7b': 'Qwen 2.5 Coder 7B',
        'llama3.1:8b': 'Llama 3.1 8B'
    }
    order = ['qwen2.5-coder:7b', 'qwen2.5:7b', 'llama3.2:1b', 'mistral:7b', 'llama3.2:3b', 'deepseek-r1:7b', 'phi3.5:3.8b', 'llama3.1:8b', 'gemma2:9b']

    for m in order:
        s = data['model_summaries'][m]
        print(f"  {display[m]:20s} {s['accuracy']:>9.1%} "
              f"{s['consistency_score']:>9.1%} {s['robustness_score']:>9.1%} "
              f"{s['fault_tolerance_score']:>9.1%} {s['safety_score']:>9.1%} "
              f"{s['composite_reliability']:>9.1%}")

    pdf = PAPER_DIR / "main.pdf"
    docx = PAPER_DIR / "small_agent_reliability.docx"
    print(f"\n  Paper: {pdf.name} ({pdf.stat().st_size//1024} KB)" if pdf.exists() else "\n  Paper: not compiled")
    print(f"  DOCX:  {docx.name} ({docx.stat().st_size//1024} KB)" if docx.exists() else "  DOCX:  not generated")
    print(f"  GitHub: https://github.com/Praansu/small-agent-reliability")
    print()

def cmd_help(args=None):
    """Print help."""
    print("""
Small Agent Reliability — Research Automation Tool

Usage: python automate.py <command> [options]

Commands:
  pull <model>     Pull a model via Ollama
  run [--model]    Run experiments (all models or specific)
  analyze          Re-analyze results + regenerate figures
  compile          Compile LaTeX paper to PDF
  docx             Generate DOCX document
  publish          Full pipeline: analyze → compile → docx → Desktop
  status           Show current research status
  help             Show this help
""")

def main():
    parser = argparse.ArgumentParser(description="Research Automation")
    parser.add_argument("command", nargs="?", default="help",
                       choices=["pull", "run", "analyze", "compile", "docx", "publish", "status", "help"])
    parser.add_argument("--model", "-m", help="Model name for pull/run")
    args = parser.parse_args()

    commands = {
        "pull": cmd_pull_model,
        "run": cmd_run_experiment,
        "analyze": cmd_analyze,
        "compile": cmd_compile,
        "docx": cmd_docx,
        "publish": cmd_publish,
        "status": cmd_status,
        "help": cmd_help,
    }

    commands[args.command](args)

if __name__ == "__main__":
    main()
