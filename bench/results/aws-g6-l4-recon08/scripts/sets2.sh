rmdir ~/br 2>/dev/null; ln -sfn ~/old ~/br
cd ~
python3 ~/mk_ll_sets.py > ~/results/setup/llsets.log 2>&1; echo "exit $?" >> ~/results/setup/llsets.log
python3 ~/mk_sets.py > ~/results/setup/sets.log 2>&1; echo "exit $?" >> ~/results/setup/sets.log
bash ~/mk_mono.sh > ~/results/setup/mono.log 2>&1; echo "exit $?" >> ~/results/setup/mono.log
for d in ~/sets/*/ ~/ll/*/ ~/pal/sets/*/; do echo "$d: $(ls $d | wc -l) files"; done
echo SETS_DONE
