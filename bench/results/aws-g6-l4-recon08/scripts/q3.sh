# q3.sh: latency check: single-file mode (8 MP, 2.5 MP, a thumbnail) and thumbnails / standard at >= 250 MP
TAG=q_lat MINMP=250 SETS=single,single2,thumbs,std python3 ~/r/spd8.py main=$HOME/old/build/av1gpu c1=$HOME/bins/c1 c3=$HOME/bins/c3
