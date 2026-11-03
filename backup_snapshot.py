"""
Backup snapshot script
======================
  data/users.json / data/keiba.db / models/*.pkl / data/.jwt_secret
  を tar.gz に固めてローテーション保持する。

運用想定:
  - cron / systemd timer から毎日呼び出す
  - 既定で backups/daily/ 配下に最大 14 世代、backups/weekly/ に最大 8 世代
  - `.jwt_secret` を含むため **出力先ディレクトリは 0700 で運用すること**。
  - 暗号化が必要な場合は --encrypt-key でファイル (password) を指定すれば
    openssl aes-256-cbc で pipe 暗号化する。

使い方:
  python3 backup_snapshot.py                    # 日次 (デフォルト)
  python3 backup_snapshot.py --kind weekly      # 週次
  python3 backup_snapshot.py --dest /mnt/vault  # 外部ディスクへ
  python3 backup_snapshot.py --dry-run          # 何がバックアップされるか確認
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("backup")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DEST = REPO_ROOT / "backups"

# バックアップ対象 — (相対パス, 必須か)
TARGETS: list[tuple[str, bool]] = [
    ("data/users.json", True),
    ("data/keiba.db", False),
    ("data/.jwt_secret", True),
    ("data/.admin_initial_password", False),
    ("data/bloodline.csv", False),
    ("data/training.csv", False),
    ("data/notification_settings.json", False),
    ("data/api_keys.json", False),
    ("data/payouts.csv", False),
    ("data/backtest_results.json", False),
    ("data/honest_backtest_results.json", False),
    ("data/frame_bias.json", False),
    ("data/odds_overlay_stats.json", False),
    ("data/auto_learn_state.json", False),
    ("models", False),  # ディレクトリまるごと
]

# ローテーション世代数
RETENTION = {
    "daily": 14,
    "weekly": 8,
    "monthly": 12,
    "manual": 0,  # 0 = 無制限 (自動削除しない)
}


def _iter_targets() -> list[Path]:
    out: list[Path] = []
    for rel, required in TARGETS:
        p = REPO_ROOT / rel
        if p.exists():
            out.append(p)
        elif required:
            raise FileNotFoundError(f"必須バックアップ対象が見つからない: {p}")
        else:
            logger.info("skip (not present): %s", rel)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_manifest(paths: list[Path]) -> dict:
    files = []
    total = 0
    for p in paths:
        if p.is_file():
            size = p.stat().st_size
            files.append({
                "path": str(p.relative_to(REPO_ROOT)),
                "size": size,
                "mtime": p.stat().st_mtime,
            })
            total += size
        elif p.is_dir():
            for sub in p.rglob("*"):
                if sub.is_file():
                    size = sub.stat().st_size
                    files.append({
                        "path": str(sub.relative_to(REPO_ROOT)),
                        "size": size,
                        "mtime": sub.stat().st_mtime,
                    })
                    total += size
    return {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "host": os.uname().nodename,
        "repo_root": str(REPO_ROOT),
        "total_bytes": total,
        "file_count": len(files),
        "files": files,
    }


def _snapshot_name(kind: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{kind}_{now.strftime('%Y%m%d_%H%M%S')}.tar.gz"


def _write_archive(paths: list[Path], out_path: Path, manifest: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write: 一時名→fsync→rename
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with tarfile.open(tmp, "w:gz") as tar:
        # manifest を先頭に
        manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        import io
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for p in paths:
            arcname = str(p.relative_to(REPO_ROOT))
            tar.add(str(p), arcname=arcname, recursive=True)
    # fsync
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    tmp.rename(out_path)
    # 権限を締める (0600)
    os.chmod(out_path, 0o600)


def _encrypt_file(src: Path, key_file: Path) -> Path:
    """openssl aes-256-cbc で暗号化。鍵は key_file に平文で保管。"""
    dst = src.with_suffix(src.suffix + ".enc")
    cmd = [
        "openssl", "enc", "-aes-256-cbc", "-salt", "-pbkdf2",
        "-in", str(src), "-out", str(dst),
        "-pass", f"file:{key_file}",
    ]
    subprocess.check_call(cmd)
    os.chmod(dst, 0o600)
    src.unlink()
    return dst


def _rotate(dest_dir: Path, kind: str) -> list[Path]:
    """古い世代を削除し、残ったファイル一覧を返す。"""
    keep = RETENTION.get(kind, 0)
    files = sorted(dest_dir.glob(f"{kind}_*.tar.gz*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if keep <= 0:
        return files
    remove = files[keep:]
    for old in remove:
        logger.info("rotate out: %s", old.name)
        old.unlink()
    return files[:keep]


def backup(kind: str, dest: Path, encrypt_key: Path | None, dry_run: bool) -> Path | None:
    paths = _iter_targets()
    manifest = _make_manifest(paths)
    archive = dest / kind / _snapshot_name(kind)

    logger.info("targets: %d files, %.2f MB", manifest["file_count"], manifest["total_bytes"] / 1e6)
    if dry_run:
        for f in manifest["files"][:20]:
            logger.info("  %s (%d bytes)", f["path"], f["size"])
        if len(manifest["files"]) > 20:
            logger.info("  ... %d more", len(manifest["files"]) - 20)
        logger.info("[DRY RUN] would write: %s", archive)
        return None

    _write_archive(paths, archive, manifest)
    logger.info("wrote: %s (%.2f MB)", archive, archive.stat().st_size / 1e6)

    if encrypt_key:
        archive = _encrypt_file(archive, encrypt_key)
        logger.info("encrypted: %s", archive.name)

    digest = _sha256(archive)
    (archive.parent / (archive.name + ".sha256")).write_text(f"{digest}  {archive.name}\n")

    _rotate(archive.parent, kind)
    return archive


def verify(archive: Path) -> bool:
    """tarball を読み出して manifest と突き合わせる。破損していないか確認。"""
    with tarfile.open(archive, "r:gz") as tar:
        try:
            mem = tar.getmember("MANIFEST.json")
        except KeyError:
            logger.error("no MANIFEST.json in %s", archive)
            return False
        f = tar.extractfile(mem)
        if f is None:
            return False
        manifest = json.load(f)
        names = {m.name for m in tar.getmembers() if m.isfile()}
    expected = {x["path"] for x in manifest["files"]} | {"MANIFEST.json"}
    missing = expected - names
    if missing:
        logger.error("missing in archive: %s", sorted(missing)[:5])
        return False
    logger.info("verify ok: %d files", len(manifest["files"]))
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="keiba-predictor backup snapshot")
    ap.add_argument("--kind", choices=sorted(RETENTION.keys()), default="daily")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--encrypt-key", type=Path, default=None,
                    help="openssl pass file (平文パスワードファイル、0600推奨)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", type=Path, help="既存archiveの整合性チェックのみ")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.verify:
        ok = verify(args.verify)
        return 0 if ok else 1

    # dest directory は 0700
    args.dest.mkdir(parents=True, exist_ok=True)
    os.chmod(args.dest, 0o700)

    try:
        backup(args.kind, args.dest, args.encrypt_key, args.dry_run)
    except FileNotFoundError as e:
        logger.error("backup failed: %s", e)
        return 2
    except Exception as e:
        logger.exception("backup failed: %s", e)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
