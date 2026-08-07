#!/usr/bin/env bash
echo "=== NODE $(hostname) === user=$(whoami) id=$(id -u):$(id -g)"
echo "--- FS types (need non-overlayfs for podman overlay driver) ---"
for d in / /tmp "$TMPDIR" /scratch /dev/shm "$HOME"; do
  [ -n "$d" ] && [ -e "$d" ] && echo "  $d -> $(stat -f -c '%T' "$d" 2>/dev/null)"
done
echo "--- container runtimes ---"
for c in podman docker apptainer singularity; do
  printf "  %-12s " "$c"; command -v "$c" >/dev/null && "$c" --version 2>&1 | head -1 || echo absent
done
printf "  userns: "; timeout 10 unshare -Ur echo OK 2>&1 | tail -1
echo "--- egress ---"
probe(){ printf "  %-40s " "$1"; timeout 12 curl -s -o /dev/null -w "HTTP %{http_code}\n" "$2" 2>&1 | tail -1; }
probe "docker.io CDN" "https://production.cloudfront.docker.com/"
probe "registry-1.docker.io/v2" "https://registry-1.docker.io/v2/"
probe "ghcr.io/v2" "https://ghcr.io/v2/"
probe "pypi" "https://pypi.org/simple/"
probe "SandoQ gateway" "https://sandoq.eks-prod.cf.aws.metafb.cloud/"
printf "  %-40s " "fwdproxy resolve"; getent hosts fwdproxy >/dev/null 2>&1 && echo YES || echo NO
printf "  %-40s " "Qwen cpu-000-198:8100 TCP"; timeout 8 bash -c ":</dev/tcp/cpu-000-198/8100" 2>/dev/null && echo OK || echo FAIL
probe "Opus x2p gateway" "http://anthropic.ai-gateway.x2p.facebook.net/"
echo "--- real container run (overlay+fuse-overlayfs, then vfs fallback) ---"
export XDG_RUNTIME_DIR=/tmp/pdm-$$; mkdir -p "$XDG_RUNTIME_DIR"
FUSE=$(command -v fuse-overlayfs 2>/dev/null)
timeout 200 podman --root /tmp/ps-ov-$$ --storage-driver overlay ${FUSE:+--storage-opt overlay.mount_program=$FUSE} \
  run --rm docker.io/library/ubuntu:24.04 echo PODMAN_OVERLAY_OK 2>&1 | tail -4
echo "  --- vfs fallback ---"
timeout 220 podman --root /tmp/ps-vfs-$$ --storage-driver vfs \
  run --rm docker.io/library/ubuntu:24.04 echo PODMAN_VFS_OK 2>&1 | tail -4
echo "=== PROBE DONE ==="
