"""Expose a checkpoint spill directory to the host workspace and a live container.

Requires Linux root, Docker and open_tree/move_mount (Linux >= 5.2). For newly
created containers, prefer a normal additional Docker bind mount instead.
Only creates a mount at WORKSPACE/.checkpoint-spill; no checkpoint is moved.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid


def run(*command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--container-workspace', default='/workspace')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'x86_64'):
        parser.error('Supported syscall ABI: Linux aarch64/x86_64')
    root = args.storage.resolve()
    workspace = args.workspace.resolve()
    if not workspace.is_dir() or root == workspace or root.is_relative_to(workspace):
        parser.error('Existing workspace and separate external storage required')
    root.mkdir(parents=True, exist_ok=True)
    target = workspace / '.checkpoint-spill'
    target.mkdir(exist_ok=True)
    if os.path.ismount(target):
        if not os.path.samefile(root, target):
            raise RuntimeError('Host mount points to another directory')
    else:
        if any(target.iterdir()):
            raise RuntimeError('Refusing to hide existing workspace files')
        subprocess.run(['mount', '--bind', str(root), str(target)], check=True)
    info = json.loads(run('docker', 'inspect', args.container))[0]
    if not info['State']['Running']:
        raise RuntimeError('Container must be running')
    python = run('docker', 'exec', args.container, 'sh', '-c', 'command -v python3')
    inside = args.container_workspace.rstrip('/') + '/.checkpoint-spill'
    probe = root / ('.mapping-probe-' + uuid.uuid4().hex)
    probe.write_text(uuid.uuid4().hex)
    try:
        check = subprocess.run(['docker', 'exec', args.container, 'cat',
                                inside + '/' + probe.name], capture_output=True, text=True)
        if check.returncode or check.stdout != probe.read_text():
            contents = json.loads(run('docker', 'exec', args.container, python, '-c',
                                      'import json,os,sys; print(json.dumps(os.listdir(sys.argv[1])))',
                                      inside))
            if contents:
                raise RuntimeError('Refusing to hide files in the container mount target')
            # The detached mount descriptor may be attached in another namespace;
            # binding a plain directory FD from a foreign namespace is invalid.
            c = ctypes.CDLL(None, use_errno=True)
            fd = c.syscall(428, -100, os.fsencode(root), 1 | os.O_CLOEXEC)
            if fd < 0:
                raise OSError(ctypes.get_errno(), 'open_tree failed')
            code = ('import ctypes,os; os.chdir("/"); '
                    'c=ctypes.CDLL(None,use_errno=True); '
                    f'r=c.syscall(429,{fd},b"",-100,{os.fsencode(inside)!r},4); '
                    'assert r==0,os.strerror(ctypes.get_errno())')
            try:
                subprocess.run(['nsenter', '-t', str(info['State']['Pid']),
                                '--mount', '--pid', '--root', '--wd=/', '--',
                                python, '-c', code], pass_fds=(fd,), check=True)
            finally:
                os.close(fd)
        observed = run('docker', 'exec', args.container, 'cat', inside + '/' + probe.name)
        if observed != probe.read_text():
            raise RuntimeError('Host/container storage mapping mismatch')
        print('Host/container storage mapping verified')
    finally:
        probe.unlink()


if __name__ == '__main__':
    main()
