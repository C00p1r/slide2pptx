"""Run a long training job on Colab so it survives the browser page.

    python tools/colab_job.py start NAME --dir RUNS_DIR -- <command ...>
    python tools/colab_job.py status NAME --dir RUNS_DIR      # running? last log lines
    python tools/colab_job.py stop NAME --dir RUNS_DIR

`start` launches the command in its own session (setsid + nohup), so closing or reloading
the notebook page, or the page losing its connection, does not kill it. Output goes to
RUNS_DIR/NAME.log (put RUNS_DIR on Drive and the log survives a runtime reset too).
Starting a job that is already running does nothing, so re-running the notebook cell is safe.

Together with `--resume auto` in boxseg.train (the default), the recovery after a lost
runtime is: reconnect, run the setup cell, run the same start cell again. Training continues
from the last checkpoint (every 250 steps), and finished stages are skipped.
Works the same from a notebook cell (`!python tools/colab_job.py ...`) and the Colab terminal.
"""
import argparse
import os
import shlex
import signal
import subprocess
import sys


def _paths(a):
    os.makedirs(a.dir, exist_ok=True)
    return os.path.join(a.dir, f"{a.name}.pid"), os.path.join(a.dir, f"{a.name}.log")


def _alive(pidfile):
    try:
        pid = int(open(pidfile).read().split()[0])
        os.kill(pid, 0)
        # pid files live on Drive and outlive the VM: make sure it is our process on this VM
        log = pidfile[:-4] + ".log"
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return pid if os.path.basename(log).encode() in f.read() else None
    except (OSError, ValueError, IndexError):
        return None


def start(a):
    pidfile, log = _paths(a)
    if _alive(pidfile):
        print(f"{a.name} is already running (pid {_alive(pidfile)}). Log: {log}")
        return
    cmd = a.cmd
    if not cmd:
        sys.exit("no command given (put it after --)")
    line = cmd[0] if len(cmd) == 1 else shlex.join(cmd)    # one string = a shell pipeline (&& allowed)
    with open(log, "a") as f:
        f.write(f"\n===== start: {line}\n")
    p = subprocess.Popen(["bash", "-c", f"( {line} ) >> {shlex.quote(log)} 2>&1; echo \"===== exit $?\" >> {shlex.quote(log)}"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    with open(pidfile, "w") as f:
        f.write(f"{p.pid}\n")
    print(f"started {a.name} (pid {p.pid}). Log: {log}")


def status(a):
    pidfile, log = _paths(a)
    pid = _alive(pidfile)
    print(f"{a.name}: {'RUNNING (pid %d)' % pid if pid else 'not running'}")
    if os.path.exists(log):
        lines = open(log, errors="replace").read().splitlines()
        print("\n".join(lines[-a.lines:]))


def stop(a):
    pidfile, _ = _paths(a)
    pid = _alive(pidfile)
    if not pid:
        print(f"{a.name} is not running")
        return
    os.killpg(pid, signal.SIGTERM)
    print(f"stopped {a.name} (pid {pid})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["start", "status", "stop"])
    ap.add_argument("name")
    ap.add_argument("--dir", default="runs/jobs")
    ap.add_argument("--lines", type=int, default=15)
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:                      # everything after -- is the job's command
        k = argv.index("--")
        argv, cmd = argv[:k], argv[k + 1:]
    a = ap.parse_args(argv)
    a.cmd = cmd
    {"start": start, "status": status, "stop": stop}[a.action](a)


if __name__ == "__main__":
    main()
