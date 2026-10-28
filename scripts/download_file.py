#!/usr/bin/env python3
"""多线程分块下载器（1MB 粒度断点续传）。

GitHub / HuggingFace 在部分网络环境下会严格限速并频繁断连。该脚本把大文件拆成
1MB 分块，每个分块独立重试，并把已完成分块写入 `.progress.json`，中断后可继续。

用法:
    python scripts/download_file.py URL OUTPUT [--connections 24]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

import requests

CHUNK = 1024 * 1024


def remote_size(url: str) -> int:
    with requests.Session() as session:
        try:
            r = session.head(url, allow_redirects=True, timeout=60)
            length = int(r.headers.get("Content-Length", 0))
            if r.ok and length:
                return length
        except requests.RequestException:
            pass
        r = session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        content_range = r.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(r.headers["Content-Length"])


class ChunkDownloader:
    def __init__(self, url: str, output: Path, connections: int = 24,
                 resume_mismatched_url: bool = False) -> None:
        self.url = url
        self.output = output
        self.connections = max(1, connections)
        self.resume_mismatched_url = resume_mismatched_url
        self.size = remote_size(url)
        self.total_chunks = (self.size + CHUNK - 1) // CHUNK
        self.checkpoint = output.with_suffix(output.suffix + ".progress.json")
        self.completed: set[int] = set()
        self.lock = threading.RLock()
        self._local = threading.local()

    def load_checkpoint(self) -> None:
        if not self.checkpoint.exists():
            return
        if not self.output.exists() or self.output.stat().st_size != self.size:
            print("断点文件存在但目标文件缺失/大小不符，重新开始", flush=True)
            return
        try:
            data = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        if data.get("size") != self.size or data.get("chunk") != CHUNK:
            print("断点文件与当前下载目标不一致，将重新开始", flush=True)
            return
        if data.get("url") != self.url:
            if not self.resume_mismatched_url:
                print("断点文件与当前下载目标不一致，将重新开始", flush=True)
                return
            print(f"断点 URL 不同但文件大小一致，按同一文件续传（旧: {data.get('url')}）", flush=True)
        self.completed = set(int(i) for i in data.get("completed", []))
        print(f"恢复断点: 已完成 {len(self.completed)}/{self.total_chunks} 块", flush=True)

    def save_checkpoint(self) -> None:
        data = {
            "url": self.url,
            "size": self.size,
            "chunk": CHUNK,
            "completed": sorted(self.completed),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = self.checkpoint.with_suffix(self.checkpoint.suffix + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.checkpoint)

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def download_chunk(self, index: int) -> None:
        start = index * CHUNK
        end = min(self.size - 1, start + CHUNK - 1)
        expected = end - start + 1
        position = start
        attempts = 0
        while position <= end:
            attempts += 1
            if attempts > 10:
                raise RuntimeError(f"分块 {index} 重试超过 10 次")
            try:
                with self._session().get(
                    self.url,
                    headers={"Range": f"bytes={position}-{end}"},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    if response.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {response.status_code}")
                    with open(self.output, "r+b") as handle:
                        handle.seek(position)
                        for piece in response.iter_content(64 * 1024):
                            if not piece:
                                continue
                            handle.write(piece)
                            position += len(piece)
                    if position <= end:
                        raise RuntimeError("连接提前结束")
            except Exception as exc:  # noqa: BLE001
                print(f"[块 {index}] {type(exc).__name__}: {exc}；"
                      f"{position - start}/{expected}，重试 {attempts}/10", flush=True)
                delay = 2 + attempts
                if "429" in str(exc):
                    # 代理镜像限流时退避更久，避免连续触发
                    delay = min(15 + attempts * 5, 60)
                time.sleep(min(delay, 60))
        with self.lock:
            self.completed.add(index)
            if len(self.completed) % 10 == 0 or len(self.completed) == self.total_chunks:
                self.save_checkpoint()
                print(f"进度: {len(self.completed)}/{self.total_chunks} 块 "
                      f"({self.size_done() / 1024 / 1024:.1f}/{self.size / 1024 / 1024:.1f} MiB)",
                      flush=True)

    def size_done(self) -> int:
        with self.lock:
            return min(self.size, len(self.completed) * CHUNK)

    def run(self) -> None:
        print(f"远端大小: {self.size / 1024 / 1024:.1f} MiB，"
              f"{self.total_chunks} 块，并发 {self.connections}", flush=True)
        if not self.output.exists() or self.output.stat().st_size != self.size:
            with open(self.output, "wb") as handle:
                handle.truncate(self.size)
        self.load_checkpoint()

        pending = [i for i in range(self.total_chunks) if i not in self.completed]
        if not pending:
            print("所有分块已完成", flush=True)
            return

        started = time.time()
        stop = threading.Event()

        def report() -> None:
            while not stop.wait(30):
                done = self.size_done()
                elapsed = max(time.time() - started, 1)
                speed = done / elapsed / 1024
                remaining = (self.size - done) / max(speed * 1024, 1)
                print(f"进度: {done / 1024 / 1024:.1f}/{self.size / 1024 / 1024:.1f} MiB "
                      f"| {speed:.1f} KB/s | 预计剩余 {remaining / 60:.1f} 分钟", flush=True)

        reporter = threading.Thread(target=report, daemon=True)
        reporter.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.connections) as pool:
                futures = {pool.submit(self.download_chunk, i): i for i in pending}
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        finally:
            stop.set()
            with self.lock:
                self.save_checkpoint()
        print(f"下载完成: {self.output} "
              f"({self.output.stat().st_size / 1024 / 1024:.1f} MiB, "
              f"{time.time() - started:.0f}s)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--connections", type=int, default=24)
    parser.add_argument("--force", action="store_true", help="删除已有文件与断点后重新下载")
    parser.add_argument("--resume-mismatched-url", action="store_true",
                        help="断点 URL 与当前不同但文件大小一致时，仍按同一文件续传（用于切换镜像）")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    downloader = ChunkDownloader(args.url, args.output, args.connections,
                                 resume_mismatched_url=args.resume_mismatched_url)
    if args.force:
        args.output.unlink(missing_ok=True)
        downloader.checkpoint.unlink(missing_ok=True)
    if args.output.exists() and args.output.stat().st_size == downloader.size:
        # 大小一致且没有断点记录时，只有 zip 完整才跳过（避免复用残缺文件）
        if not downloader.checkpoint.exists() and zipfile.is_zipfile(args.output):
            print("文件已存在且 zip 完整，跳过下载")
            return 0
    try:
        downloader.run()
    except KeyboardInterrupt:
        print("\n已中断，进度已保存，可重新运行续传", file=sys.stderr)
        return 130

    if zipfile.is_zipfile(args.output):
        with zipfile.ZipFile(args.output) as zf:
            bad = zf.testzip()
        print("zip 校验通过" if bad is None else f"zip 校验失败: {bad}", flush=True)
        return 0 if bad is None else 2
    print("下载完成但不是有效 zip（若为其他格式请忽略）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
