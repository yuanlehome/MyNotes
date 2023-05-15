# git撤销提交（已经push到远程）
git reset HEAD^
git stash
git reset --hard upstream/develop
git push -f
git stash pop

# git设置远程分支追踪
git branch --set-upstream-to=origin/develop

# git设置远程仓库地址
git remote set-url origin https://github.com/yuanlehome/Paddle.git
git remote add upstream https://github.com/PaddlePaddle/Paddle.git

# git clone特定tag
git clone <repo_url> --branch <tag_name> --single-branch
