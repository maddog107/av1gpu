# pack.sh: the result logs for the repository (small text only) into ~/recon08_logs.tgz
cd ~
mkdir -p ~/pack/logs/runlogs ~/pack/logs/setup
cp -r ~/results/race ~/results/memcheck ~/results/synccheck ~/results/resusage_full ~/results/spd8 ~/pack/logs/
cp ~/results/*.txt ~/results/final.log ~/results/gate_*.log ~/pack/logs/
cp ~/r/*.log ~/pack/logs/runlogs/ 2>/dev/null
for f in ~/results/setup/*.log; do case $f in *apt.log) ;; *) cp $f ~/pack/logs/setup/ ;; esac; done
rm -f ~/pack/logs/identity.txt ~/pack/logs/env.txt   # fetched before / recorded locally
tar -czf ~/recon08_logs.tgz -C ~/pack logs
ls -la ~/recon08_logs.tgz
