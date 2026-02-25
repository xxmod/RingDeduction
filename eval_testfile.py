#!/usr/bin/env python3
"""
评估 testfile 文件夹内截图的 Top1 / Top5 正确率。

默认读取：
- 截图目录: testfile/
- 标注文件: testfile/test-elevant.txt

用法示例:
    python eval_testfile.py
    python eval_testfile.py --dir testfile --labels testfile/test-elevant.txt
    python eval_testfile.py --quiet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import main


def parse_labels(label_path: Path) -> List[Tuple[str, str, str]]:
    """解析标注文件，返回 [(map_name, screenshot_stem, frame_name.jpg), ...]。"""
    if not label_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {label_path}")

    lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    current_map = None
    items: List[Tuple[str, str, str]] = []

    for line in lines:
        if "->" not in line:
            current_map = line
            continue
        if not current_map:
            continue

        left, right = [x.strip() for x in line.split("->", 1)]
        frame_name = right if right.lower().endswith(".jpg") else f"{right}.jpg"
        items.append((current_map, left, frame_name))

    return items


def load_caches(items: List[Tuple[str, str, str]], root: Path) -> Dict[str, dict]:
    """按标注中出现的地图加载对应 cache。"""
    maps = sorted({map_name for map_name, _, _ in items})
    caches: Dict[str, dict] = {}

    for map_name in maps:
        cache_path = root / f"{map_name}-cache.json"
        if not cache_path.exists():
            raise FileNotFoundError(f"缓存文件不存在: {cache_path}")
        caches[map_name] = json.loads(cache_path.read_text(encoding="utf-8"))

    return caches


def predict_top5(img_path: Path, cache: dict) -> List[Tuple[str, float]]:
    """对单张截图返回 Top5 结果（尽量复用主流程匹配策略）。"""
    img = main.imread_safe(img_path)
    if img is None:
        return []

    map_img = main.crop_map_from_screenshot(img)
    results = main.match_by_overlay(map_img, cache)

    if not results:
        det = main.detect_ring_screenshot(map_img)
        if det is None:
            return []
        ring_det, _ = det
        results = main.match_ring(ring_det, map_img.shape, cache)

    return results


def run_eval(test_dir: Path, labels: Path, quiet: bool = False) -> int:
    if quiet:
        main.log = lambda *args, **kwargs: None

    items = parse_labels(labels)
    if not items:
        print("[ERROR] 标注文件为空或格式无效")
        return 1

    root = Path(__file__).resolve().parent
    caches = load_caches(items, root)

    total = 0
    hit_top1 = 0
    hit_top5 = 0
    missing_files = 0
    no_prediction = 0

    for map_name, screenshot_stem, gt in items:
        img_path = test_dir / f"{screenshot_stem}.png"
        if not img_path.exists():
            missing_files += 1
            if not quiet:
                print(f"[MISS] 文件不存在: {img_path.name}")
            continue

        total += 1
        results = predict_top5(img_path, caches[map_name])
        if not results:
            no_prediction += 1
            if not quiet:
                print(f"[FAIL] 无法预测: {img_path.name} | GT={gt}")
            continue

        top_names = [name for name, _score in results]
        pred = top_names[0]
        top1_ok = pred == gt
        top5_ok = gt in top_names

        hit_top1 += int(top1_ok)
        hit_top5 += int(top5_ok)

        if not quiet:
            print(
                f"[{map_name}] {img_path.name} | GT={gt} | "
                f"P1={pred} | H1={top1_ok} | H5={top5_ok}"
            )

    print("\n" + "=" * 64)
    print(f"可评估样本数: {total}")
    print(f"Top1 命中: {hit_top1}/{total} ({(hit_top1 / total * 100) if total else 0:.2f}%)")
    print(f"Top5 命中: {hit_top5}/{total} ({(hit_top5 / total * 100) if total else 0:.2f}%)")
    print(f"缺失截图文件: {missing_files}")
    print(f"无预测结果: {no_prediction}")
    print("=" * 64)

    return 0


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="评估 testfile 截图 Top1/Top5 正确率")
    parser.add_argument("--dir", default="testfile", help="测试截图目录（默认: testfile）")
    parser.add_argument("--labels", default="testfile/test-elevant.txt", help="标注文件路径")
    parser.add_argument("--quiet", action="store_true", help="仅输出汇总")
    args = parser.parse_args()

    test_dir = Path(args.dir)
    labels = Path(args.labels)
    return run_eval(test_dir=test_dir, labels=labels, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main_cli())
