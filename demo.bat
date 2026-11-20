@echo off
call .venv\Scripts\activate
set /p IMG=Drag the image here and press Enter: 
python -m src.pipeline --image %IMG% --weights runs\detect\baseline\weights\best.pt --conf 0.25 --heading 45
start outputs\annotated.png
type outputs\report.csv
pause