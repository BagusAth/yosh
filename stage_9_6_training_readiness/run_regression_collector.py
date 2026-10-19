import subprocess
import sys
from pathlib import Path

tests = [
    ("test_stage_9_3_regression.py", "Stage 9.3 Regression Suite"),
    ("test_stage_9_4_regression.py", "Stage 9.4 Regression Suite"),
    ("test_stage_9_5_regression.py", "Stage 9.5 Regression Suite"),
    ("test_stage_9_6_regression.py", "Stage 9.6 Regression Suite"),
]

out_lines = []
out_lines.append("================================================================================")
out_lines.append("STAGE 9.6 COMPLETE REGRESSION TEST SUITE EXECUTION LOG")
out_lines.append("================================================================================\n")

all_passed = True

for script, name in tests:
    out_lines.append(f">>> RUNNING: {name} ({script})")
    p = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.stdout:
        out_lines.append(p.stdout)
    if p.stderr:
        out_lines.append(p.stderr)
    out_lines.append(f"Exit Code: {p.returncode}")
    if p.returncode != 0:
        all_passed = False
        out_lines.append("STATUS: FAILED\n")
    else:
        out_lines.append("STATUS: PASSED\n")
    out_lines.append("-" * 80 + "\n")

out_lines.append(f"OVERALL STATUS: {'ALL PASSED (100% OK)' if all_passed else 'FAILURES DETECTED'}")
out_p = Path("stage_9_6_training_readiness/regression_test_results.txt")
out_p.write_text("\n".join(out_lines), encoding="utf-8")
print(f"Results written to {out_p}. Overall status: {'PASSED' if all_passed else 'FAILED'}")
