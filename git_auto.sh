now=$(date "+%Y-%m-%d")
echo "Change Directory to C:/my_github"
cd C:/my_github || { echo "Error: 目录不存在，退出脚本"; read; exit 1; }

echo "Starting add-commit-pull-push..."

# 先拉取远程最新代码，强制覆盖本地冲突文件
echo "拉取远程最新代码..."
git pull origin main --rebase || {
    echo "拉取代码冲突，尝试强制覆盖本地文件..."
    git fetch --all
    git reset --hard origin/main
}

# 添加所有修改并提交
git add -A
git commit -m "Auto commit: $now" || {
    echo "没有需要提交的修改，直接推送..."
}

# 推送本地修改到远程
git push origin main || {
    echo "推送失败，尝试强制推送..."
    git push origin main -f
}

echo "Finish!"
read
