@echo off
cd /d "%~dp0"
echo Berhenti melacak file prediksi per citra di git (file TETAP ada di disk)...
git rm -r --cached --quiet --ignore-unmatch -- stage_v2_ablation/per_image/_lama_nms_terpotong stage_v2_eleak_sim/per_image/preds_group_on_cleanval.csv stage_v2_eleak_sim/per_image/preds_sim_on_simval.csv stage_v2_eleak_sim/per_image/gt_cleanval.csv stage_v2_eleak_sim/per_image/gt_simval.csv stage_v2_eleak_sim/per_image/analysis_cache.npz
git add .gitignore rapikan_git.bat
git commit -m "Berhenti melacak prediksi per citra (file tetap ada di disk)" --trailer "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" --trailer "Claude-Session: https://claude.ai/code/session_015oXrwUQZdycPnPqBtm95qT"
echo.
git status --short
echo Selesai.
pause
