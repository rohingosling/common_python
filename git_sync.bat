cls
echo off
sound 12000 50

echo -
echo - GIT Status.
echo -
git status

sound 12000 50
echo -
echo - Adding files.
echo -
git add *

sound 12000 50
echo - GIT Status.
echo -
git status
echo

sound 12000 50
echo Commiting changes.
echo -
git commit -m "Auto sync code."

sound 12000 50
echo -
echo - GIT Status.
echo -
git status

sound 12000 50
echo -
echo - Pushing master to origin.
echo -
git push origin master

sound 12000 50
echo -
echo - GIT Status.
echo -
git status
sound 12000 50
echo -
echo - Active branch.
echo -
git branch
sound 12000 50
echo on