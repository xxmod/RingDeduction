#!/usr/bin/env python3
"""
测试匹配准确度的脚本：逐个测试截图，输出 Top5 结果对照期望答案。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from main import *

# 测试用例: (截图文件名, 期望匹配的参考图, 是否确信)
TEST_CASES = [
    ("1172470_20260224111923_1.png", "frame_0191.jpg", True),
    ("1172470_20260224113702_1.png", "frame_0201.jpg", True),
    ("1172470_20260224115635_1.png", "frame_0275.jpg", True),
    ("1172470_20260224145118_1.png", "frame_0298.jpg", False),  # 疑似
    ("1172470_20260224150647_1.png", "frame_0301.jpg", False),  # 疑似
    ("1172470_20260224152751_1.png", "frame_0205.jpg", True),
]


def run_tests():
    cache = load_cache()
    results_summary = []

    for ss_name, expected, confident in TEST_CASES:
        ss_path = SCREENSHOT_DIR / ss_name
        if not ss_path.exists():
            print(f"\n{'='*60}")
            print(f"截图不存在: {ss_name}")
            results_summary.append((ss_name, expected, "MISSING", -1))
            continue

        print(f"\n{'='*60}")
        print(f"测试: {ss_name}")
        print(f"期望: {expected} {'(疑似)' if not confident else '(确信)'}")
        print(f"{'='*60}")

        ss_img = imread_safe(ss_path)
        if ss_img is None:
            results_summary.append((ss_name, expected, "READ_FAIL", -1))
            continue

        map_img = crop_map_from_screenshot(ss_img)

        # 叠加匹配
        overlay_results = match_by_overlay(map_img, cache)

        if overlay_results:
            results = overlay_results
            method = "overlay"
        else:
            det = detect_ring_screenshot(map_img)
            if det is None:
                results_summary.append((ss_name, expected, "DETECT_FAIL", -1))
                continue
            ring_det, _ = det
            results = match_ring(ring_det, map_img.shape, cache)
            method = "fallback"

        # 检查期望结果排名
        rank = -1
        for i, (name, score) in enumerate(results):
            if name == expected:
                rank = i + 1
                break

        status = "PASS" if rank == 1 else f"RANK={rank}" if rank > 0 else "NOT_IN_TOP5"
        results_summary.append((ss_name, expected, status, rank))

        print(f"方法: {method}")
        print(f"结果排名: {status}")
        print(f"Top 5:")
        for i, (name, score) in enumerate(results[:5], 1):
            marker = " <-- 期望" if name == expected else ""
            print(f"  #{i}: {name}  得分={score:.4f}{marker}")

    # 总结
    print(f"\n\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    passed = 0
    total = len(TEST_CASES)
    for ss_name, expected, status, rank in results_summary:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {ss_name} -> {expected}: {status}")
        if status == "PASS":
            passed += 1
    print(f"\n通过: {passed}/{total}")


if __name__ == "__main__":
    run_tests()
