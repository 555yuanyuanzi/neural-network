now=$(date "+%Y-%m-%d")
echo "Change Directory to C:/my_github"
cd C:/my_github
echo "Starting add-commit-pull-push..."
git add -A && git commit -m "$now" && git pull && git push
echo "Finish!"
read
