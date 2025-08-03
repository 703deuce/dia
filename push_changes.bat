@echo off
echo Pushing Docker fixes to GitHub...
git add .
git commit -m "Fix Dockerfile for RunPod build - Added dia module and dockerignore"
git push
echo Done!