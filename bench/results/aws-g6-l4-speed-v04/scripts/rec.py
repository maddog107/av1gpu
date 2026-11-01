# run each command in a shell, record (seconds since start of the command, line) for the GIF
import json, subprocess, sys, time, os
cmds = json.loads(sys.argv[1])
env = dict(os.environ, PATH=os.path.expanduser("~/av1gpu-readme/av1gpu/build") + ":" + os.environ["PATH"])
out = []
for c in cmds:
    t0 = time.time()
    p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         cwd=os.path.expanduser("~/av1gpu-readme"), env=env)
    lines = [(round(time.time() - t0, 3), l.rstrip("\n")) for l in p.stdout]
    p.wait()
    out.append(dict(cmd=c, lines=lines, total=round(time.time() - t0, 3)))
print(json.dumps(out))
