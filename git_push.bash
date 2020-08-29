git pull
find ./ -iname "*checkpoint*" | xargs rm -rf
git del *.pyc
git add utils
git add notebooks
git add train_scripts
git add *.bash
git add *.sh
git add *.yml
git add *.py
