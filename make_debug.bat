@echo off
set jsonpath=egs/debug/tr
if not exist %jsonpath% then mkdir %jsonpath%
python -m denoiser.audio dataset/debug/noisy - %jsonpath%/noisy.json
python -m denoiser.audio dataset/debug/clean - %jsonpath%/clean.json