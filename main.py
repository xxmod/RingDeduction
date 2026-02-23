#!/usr/bin/env python3
"""
Ring Deduction - Apex Legends 缩圈预测工具 (Broken Moon)

从游戏截图中检测当前圈的位置和大小，
在预生成的参考图库中找到最匹配的圈配置图并打开，
从而预测后续缩圈走向。

用法:
    python main.py              # 正常运行
    python main.py --rebuild    # 强制重建参考图缓存
    python main.py --no-debug   # 关闭调试可视化
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from pathlib import Path

# ======================== 配置 ========================
SCREENSHOT_DIR = Path("./screenshot")
MAP_DIR = Path("./map/broken-moon")
CACHE_FILE = Path("./ring_cache.json")
DEBUG_DIR = Path("./debug")

TOP_N = 5        # 显示前 N 个最佳匹配
DEBUG = True      # 是否保存调试图


# ======================== 工具函数 ========================

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


def get_latest_screenshot():
    """选取 screenshot 目录中创建时间最新的图片。"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    images = [f for f in SCREENSHOT_DIR.iterdir() if f.suffix.lower() in exts]
    if not images:
        log(f"在 {SCREENSHOT_DIR} 中未找到图片文件", "ERROR")
        sys.exit(1)
    latest = max(images, key=lambda p: p.stat().st_ctime)
    log(f"选取最新截图: {latest.name}")
    return latest


# ======================== 图像读取（支持中文路径） ========================

def imread_safe(filepath):
    """使用 np.fromfile + cv2.imdecode 读取图片，支持中文路径。"""
    data = np.fromfile(str(filepath), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# ======================== 地图裁剪 ========================

# 从游戏截图中裁剪地图区域的比例 (left, top, right, bottom)
# 适用于 16:9 分辨率 (1920×1080 等)
SCREENSHOT_MAP_CROP = (0.255, 0.035, 0.745, 0.905)


def crop_map_from_screenshot(img):
    """
    从 Apex Legends 全屏地图截图中提取地图区域。
    使用固定比例裁剪，去除左侧挑战面板、右下图例面板、底部操作栏等 UI。
    """
    h, w = img.shape[:2]
    l, t, r, b = SCREENSHOT_MAP_CROP
    cropped = img[int(h * t):int(h * b), int(w * l):int(w * r)]
    log(f"裁剪: ({int(w*l)},{int(h*t)})-({int(w*r)},{int(h*b)}) -> {cropped.shape[1]}×{cropped.shape[0]}")
    return cropped


# ======================== 圆检测核心 ========================

def _hough(gray, min_r, max_r, dp, p1, p2, blur_k=0):
    """执行一次 HoughCircles。"""
    src = cv2.GaussianBlur(gray, (blur_k, blur_k), 2) if blur_k > 2 else gray
    res = cv2.HoughCircles(
        src, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=max(min_r, 30),
        param1=p1, param2=p2,
        minRadius=min_r, maxRadius=max_r,
    )
    if res is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in res[0]]


def _cluster(circles, tol):
    """聚类相近的圆，返回列表按 (投票数, 半径) 降序。"""
    if not circles:
        return []
    pts = np.array(circles)
    used = np.zeros(len(pts), dtype=bool)
    out = []
    for i in range(len(pts)):
        if used[i]:
            continue
        grp = [pts[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            if (np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < tol
                    and abs(pts[i][2] - pts[j][2]) < tol):
                grp.append(pts[j])
                used[j] = True
        m = np.mean(grp, axis=0)
        out.append((m[0], m[1], m[2], len(grp)))
    out.sort(key=lambda x: (x[3], x[2]), reverse=True)
    return [(o[0], o[1], o[2]) for o in out]


def _detect_crosshair(gray, threshold=170, coverage=0.35):
    """
    检测截图中的十字线（鼠标路径标记）。
    十字线是跨越大半个图像的水平/垂直亮线。
    返回 (crosshair_rows, crosshair_cols) — 需要遮蔽的行和列索引列表。
    """
    h, w = gray.shape
    rows = []
    cols = []

    # 检测水平亮线
    for y in range(h):
        bright = np.sum(gray[y, :] > threshold)
        if bright > w * coverage:
            rows.append(y)

    # 检测垂直亮线
    for x in range(w):
        bright = np.sum(gray[:, x] > threshold)
        if bright > h * coverage:
            cols.append(x)

    return rows, cols


def _mask_crosshair(gray, rows, cols, width=5):
    """
    将十字线区域替换为周围像素的中值，消除干扰。
    """
    cleaned = gray.copy()
    h, w = gray.shape

    for y in rows:
        y_lo = max(0, y - width)
        y_hi = min(h, y + width + 1)
        # 用十字线上下方的像素中值填充
        above = max(0, y - width * 2)
        below = min(h, y + width * 2 + 1)
        for x in range(w):
            neighbors = []
            if above < y_lo:
                neighbors.extend(gray[above:y_lo, x].tolist())
            if y_hi < below:
                neighbors.extend(gray[y_hi:below, x].tolist())
            if neighbors:
                cleaned[y_lo:y_hi, x] = int(np.median(neighbors))

    for x in cols:
        x_lo = max(0, x - width)
        x_hi = min(w, x + width + 1)
        left = max(0, x - width * 2)
        right = min(w, x + width * 2 + 1)
        for y in range(h):
            neighbors = []
            if left < x_lo:
                neighbors.extend(gray[y, left:x_lo].tolist())
            if x_hi < right:
                neighbors.extend(gray[y, x_hi:right].tolist())
            if neighbors:
                cleaned[y, x_lo:x_hi] = int(np.median(neighbors))

    return cleaned


def _ring_score(gray, cx, cy, r, crosshair_rows=None, crosshair_cols=None):
    """
    综合评估候选圆是否是真正的缩圈。
    评分维度：
    1. 圈内外亮度差（缩圈外有灰色蒙版变暗，仅正值时生效）
    2. 圈边界白色像素比例（排除十字线干扰）——最重要的特征
    3. 半径合理性（轻微加分，不强烈偏好大圈）
    4. 圆心居中度
    """
    h, w = gray.shape
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    # --- 1) 圈内外亮度差 ---
    inner_mask = (dist > r * 0.3) & (dist < r * 0.85)
    outer_mask = (dist > r * 1.1) & (dist < r * 1.6)

    inner_count = np.sum(inner_mask)
    outer_count = np.sum(outer_mask)
    if inner_count < 50 or outer_count < 50:
        return -100.0

    inner_mean = float(np.mean(gray[inner_mask]))
    outer_mean = float(np.mean(gray[outer_mask]))
    brightness_diff = inner_mean - outer_mean
    # 仅正亮度差有效（圈内亮于圈外），负值视为0
    brightness_score = max(0.0, brightness_diff)

    # --- 2) 圈边界白色像素比例 ---
    ring_mask = (dist > r * 0.93) & (dist < r * 1.07)

    # 排除十字线区域的干扰
    if crosshair_rows or crosshair_cols:
        exclude = np.zeros((h, w), dtype=bool)
        for row_y in (crosshair_rows or []):
            for dy in range(-5, 6):
                ry = row_y + dy
                if 0 <= ry < h:
                    exclude[ry, :] = True
        for col_x in (crosshair_cols or []):
            for dx in range(-5, 6):
                rx = col_x + dx
                if 0 <= rx < w:
                    exclude[:, rx] = True
        ring_mask = ring_mask & (~exclude)

    ring_count = np.sum(ring_mask)
    if ring_count > 0:
        white_ratio = float(np.sum(gray[ring_mask] > 200)) / ring_count
    else:
        white_ratio = 0.0

    # --- 3) 半径合理性（轻微加分，不强烈偏好大圈）---
    d_min = min(h, w)
    r_ratio = r / d_min
    if 0.20 <= r_ratio <= 0.55:
        size_bonus = 5.0  # 固定小加分，不随半径增大
    elif 0.15 <= r_ratio <= 0.62:
        size_bonus = 2.0
    else:
        size_bonus = -10.0

    # --- 4) 圆心居中度 ---
    img_cx, img_cy = w / 2, h / 2
    center_dist = np.hypot(cx - img_cx, cy - img_cy)
    center_dist_ratio = center_dist / np.hypot(img_cx, img_cy)
    center_bonus = max(0.0, 8.0 * (1.0 - center_dist_ratio * 2.0))

    # --- 综合评分 ---
    score = brightness_score + white_ratio * 100.0 + size_bonus + center_bonus
    return score


def detect_ring_screenshot(map_img):
    """
    从裁剪后的地图区域检测"大圈"（当前缩圈边界）。

    检测策略：
    1. 多组参数 HoughCircles 收集候选圆
    2. 过滤掉明显不合理的圆
    3. 聚类去重
    4. 用圈内外亮度差评分，选择最可能是缩圈的候选
    """
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    d = min(h, w)
    min_r = int(d * 0.15)
    max_r = int(d * 0.60)

    # --- 检测并去除十字线干扰 ---
    ch_rows, ch_cols = _detect_crosshair(gray)
    if ch_rows or ch_cols:
        log(f"检测到十字线: {len(ch_rows)}条水平, {len(ch_cols)}条垂直", "DEBUG")
        gray_clean = _mask_crosshair(gray, ch_rows, ch_cols)
    else:
        gray_clean = gray

    candidates = []

    # 策略1: 重度模糊 + Canny 边缘（圈内外亮度差产生边缘）
    for blur_k in [31, 51, 71]:
        heavy = cv2.GaussianBlur(gray_clean, (blur_k, blur_k), 0)
        edges = cv2.Canny(heavy, 15, 50)
        for dp, p2 in [(1.5, 12), (2.0, 12), (1.2, 15)]:
            candidates.extend(_hough(edges, min_r, max_r, dp, 50, p2))

    # 策略2: 白色阈值（缩圈边线是白色）
    for thr in [190, 210, 230]:
        _, mask = cv2.threshold(gray_clean, thr, 255, cv2.THRESH_BINARY)
        for dp, p2 in [(1.5, 18), (2.0, 15), (1.2, 20)]:
            candidates.extend(_hough(mask, min_r, max_r, dp, 50, p2, blur_k=9))

    # 策略3: 标准灰度
    for dp, p1, p2 in [(1.0, 100, 40), (1.2, 100, 40), (1.5, 80, 30),
                        (1.2, 80, 30), (1.0, 60, 25)]:
        candidates.extend(_hough(gray_clean, min_r, max_r, dp, p1, p2, blur_k=9))

    if not candidates:
        return None

    log(f"原始候选: {len(candidates)} 个", "DEBUG")

    # 过滤：圆心在图像中央 80% 区域，圆的 >50% 在图像内
    valid = []
    for cx, cy, r in candidates:
        if cx < w * 0.1 or cx > w * 0.9:
            continue
        if cy < h * 0.1 or cy > h * 0.9:
            continue
        vis_x = (min(cx + r, w) - max(cx - r, 0)) / (2 * r) if r > 0 else 0
        vis_y = (min(cy + r, h) - max(cy - r, 0)) / (2 * r) if r > 0 else 0
        if vis_x < 0.5 or vis_y < 0.5:
            continue
        valid.append((cx, cy, r))

    log(f"过滤后: {len(valid)} 个", "DEBUG")
    if not valid:
        valid = candidates

    clusters = _cluster(valid, d * 0.04)
    if not clusters:
        return None

    # ---- 关键改进：用综合评分选择真正的缩圈 ----
    # 排除十字线干扰，综合亮度差+白色边线+尺寸+居中度
    scored = []
    for i, (cx, cy, r) in enumerate(clusters[:15]):
        s = _ring_score(gray, cx, cy, r, ch_rows, ch_cols)
        scored.append((cx, cy, r, s, i))

    # 按亮度差评分降序排列
    scored.sort(key=lambda x: x[3], reverse=True)

    for i, (cx, cy, r, bs, idx) in enumerate(scored[:5]):
        log(f"  候选#{i+1}: center=({cx:.0f},{cy:.0f}) r={r:.0f} "
            f"评分={bs:.1f} (原簇#{idx+1})", "DEBUG")

    best = scored[0]
    log(f"选择评分最高的圈: center=({best[0]:.0f},{best[1]:.0f}) "
        f"r={best[2]:.0f} 评分={best[3]:.1f}", "DEBUG")
    return (best[0], best[1], best[2]), scored


def detect_ring_reference(img):
    """检测参考图中最大的白色圆。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    d = min(h, w)
    min_r, max_r = int(d * 0.08), int(d * 0.72)

    candidates = []
    for dp, p1, p2, bk in [(1.0, 100, 40, 9), (1.2, 100, 40, 9), (1.5, 100, 50, 9)]:
        candidates.extend(_hough(gray, min_r, max_r, dp, p1, p2, bk))

    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    for dp, p2, bk in [(1.2, 25, 9), (1.5, 20, 9)]:
        candidates.extend(_hough(mask, min_r, max_r, dp, 50, p2, bk))

    if not candidates:
        return None

    valid = [(cx, cy, r) for cx, cy, r in candidates
             if w * 0.05 < cx < w * 0.95 and h * 0.05 < cy < h * 0.95]
    clusters = _cluster(valid or candidates, d * 0.04)
    return clusters[0] if clusters else None


# ======================== 缓存 ========================

def build_cache():
    files = sorted(MAP_DIR.glob("frame_*.jpg"))
    if not files:
        log(f"在 {MAP_DIR} 中未找到参考图", "ERROR")
        sys.exit(1)

    log(f"正在构建缓存，共 {len(files)} 张参考图...")
    cache = {}
    fail = 0
    for i, fp in enumerate(files):
        img = imread_safe(fp)
        if img is None:
            fail += 1; continue
        c = detect_ring_reference(img)
        if c is None:
            fail += 1; continue
        ih, iw = img.shape[:2]
        cache[fp.name] = {
            "cx": round(c[0], 2), "cy": round(c[1], 2), "r": round(c[2], 2),
            "w": iw, "h": ih,
            "cx_n": round(c[0] / iw, 5),
            "cy_n": round(c[1] / ih, 5),
            "r_n": round(c[2] / min(ih, iw), 5),
        }
        if (i + 1) % 50 == 0:
            log(f"  进度: {i+1}/{len(files)}")

    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    log(f"缓存已保存: {len(cache)} 成功, {fail} 失败")
    return cache


def load_cache(force_rebuild=False):
    if not force_rebuild and CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        log(f"已加载缓存: {len(cache)} 条记录")
        return cache
    return build_cache()


# ======================== 匹配 ========================

def match_ring(sc_circle, sc_shape, cache):
    """归一化坐标比较，返回 Top-N 匹配。"""
    h, w = sc_shape[:2]
    d = min(h, w)
    sc_cx_n = sc_circle[0] / w
    sc_cy_n = sc_circle[1] / h
    sc_r_n  = sc_circle[2] / d

    scores = []
    for name, data in cache.items():
        if "cx_n" not in data:
            continue  # 跳过元数据条目
        dx = sc_cx_n - data["cx_n"]
        dy = sc_cy_n - data["cy_n"]
        dr = sc_r_n  - data["r_n"]
        score = np.hypot(dx, dy) + abs(dr) * 0.5
        scores.append((name, score))

    scores.sort(key=lambda x: x[1])
    return scores[:TOP_N]


def match_by_overlay(map_img, cache):
    """
    叠加匹配：将缓存中每个参考圈投影到截图上，
    通过圈内外亮度对比差来找到最佳匹配。

    核心原理：Apex 中缩圈外的区域会有灰暗蒙版叠加，
    导致圈内区域比圈外区域明显更亮。将参考圈位置投影到截图上，
    计算"圈内平均亮度 - 圈外平均亮度"，值最大的就是匹配。

    这种方法不依赖 HoughCircles，不需要检测白色环线，
    而是利用缩圈的核心视觉特征（内外亮度差）来匹配。
    """
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    d = min(h, w)

    # 十字线处理
    ch_rows, ch_cols = _detect_crosshair(gray)
    if ch_rows or ch_cols:
        log(f"叠加匹配: 检测到十字线 ({len(ch_rows)}H, {len(ch_cols)}V)", "DEBUG")
        gray = _mask_crosshair(gray, ch_rows, ch_cols)

    # 重度平滑，消除地图纹理/图标/文字的干扰，保留大面积亮度差
    smooth = cv2.GaussianBlur(gray, (15, 15), 5).astype(np.float32)

    results = []
    for name, data in cache.items():
        if "cx_n" not in data:
            continue

        # 将参考圈归一化坐标映射到截图像素坐标
        cx = data["cx_n"] * w
        cy = data["cy_n"] * h
        r = data["r_n"] * d

        if r < 20:
            continue

        # 局部区域 (bounding box + margin)
        margin = r * 1.20
        y0 = max(0, int(cy - margin))
        y1 = min(h, int(cy + margin))
        x0 = max(0, int(cx - margin))
        x1 = min(w, int(cx + margin))
        if y1 - y0 < 20 or x1 - x0 < 20:
            continue

        # 局部距离平方 (使用 broadcasting)
        ly = np.arange(y0, y1, dtype=np.float32).reshape(-1, 1)
        lx = np.arange(x0, x1, dtype=np.float32).reshape(1, -1)
        dsq = (lx - cx) ** 2 + (ly - cy) ** 2

        # 圈内 (0.85r ~ 0.97r): 环状区域，避免中心点
        inner_mask = (dsq >= (r * 0.85) ** 2) & (dsq <= (r * 0.97) ** 2)
        # 圈外 (1.03r ~ 1.15r): 紧贴圈外的区域
        outer_mask = (dsq >= (r * 1.03) ** 2) & (dsq <= (r * 1.15) ** 2)

        n_in = int(np.sum(inner_mask))
        n_out = int(np.sum(outer_mask))
        if n_in < 50 or n_out < 50:
            continue

        local_smooth = smooth[y0:y1, x0:x1]
        inner_mean = float(np.mean(local_smooth[inner_mask]))
        outer_mean = float(np.mean(local_smooth[outer_mask]))

        # 核心评分: 圈内外亮度差 (正值=圈内更亮=缩圈效果)
        contrast = inner_mean - outer_mean
        results.append((name, contrast))

    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        log(f"叠加匹配 Top 5:", "DEBUG")
        for i, (nm, sc) in enumerate(results[:5]):
            dd = cache[nm]
            log(f"  #{i+1}: {nm} 对比度={sc:.1f} "
                f"cx={dd['cx_n']:.3f} cy={dd['cy_n']:.3f} r={dd['r_n']:.3f}", "DEBUG")

    return results[:TOP_N]


# ======================== 可视化 ========================

def save_debug(map_img, circle, all_candidates=None, filename="detected_ring.jpg"):
    DEBUG_DIR.mkdir(exist_ok=True)
    vis = map_img.copy()
    h, w = vis.shape[:2]

    # 画所有候选（蓝色，细线），帮助调试
    if all_candidates:
        colors = [(255, 200, 0), (200, 200, 0), (150, 150, 0),
                  (100, 100, 0), (80, 80, 0)]
        for i, item in enumerate(all_candidates[:5]):
            cx_c, cy_c, r_c, sc = item[0], item[1], item[2], item[3]
            col = colors[i] if i < len(colors) else (80, 80, 0)
            cv2.circle(vis, (int(cx_c), int(cy_c)), int(r_c), col, 2)
            cv2.putText(vis, f"#{i+1} s={sc:.1f}", (int(cx_c)-50, int(cy_c)-int(r_c)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # 画选中的圈（绿色，粗线）
    cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 3)
    cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(vis, f"SELECTED c=({cx},{cy}) r={r}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(DEBUG_DIR / filename), vis)
    log(f"调试图 -> {DEBUG_DIR / filename}")


def save_comparison(map_img, sc_circle, best_name, cache):
    DEBUG_DIR.mkdir(exist_ok=True)
    vis = map_img.copy()
    cx, cy, r = int(sc_circle[0]), int(sc_circle[1]), int(sc_circle[2])
    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 3)
    cv2.putText(vis, "Screenshot", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    ref = imread_safe(MAP_DIR / best_name)
    if ref is not None:
        d = cache[best_name]
        cv2.circle(ref, (int(d["cx"]), int(d["cy"])), int(d["r"]), (0, 255, 0), 3)
        cv2.putText(ref, best_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        th = 600
        s1 = cv2.resize(vis, (int(vis.shape[1] * th / vis.shape[0]), th))
        s2 = cv2.resize(ref, (int(ref.shape[1] * th / ref.shape[0]), th))
        cv2.imwrite(str(DEBUG_DIR / "comparison.jpg"), np.hstack([s1, s2]))
        log(f"对比图 -> {DEBUG_DIR / 'comparison.jpg'}")


# ======================== 主流程 ========================

def main():
    t0 = time.time()
    force_rebuild = "--rebuild" in sys.argv
    global DEBUG
    if "--no-debug" in sys.argv:
        DEBUG = False

    # 1. 最新截图（或指定文件）
    if "--file" in sys.argv:
        idx = sys.argv.index("--file") + 1
        if idx < len(sys.argv):
            ss_path = Path(sys.argv[idx])
            log(f"指定截图: {ss_path.name}")
        else:
            log("--file 需要指定文件路径", "ERROR"); sys.exit(1)
    else:
        ss_path = get_latest_screenshot()
    ss_img = imread_safe(ss_path)
    if ss_img is None:
        log(f"无法读取: {ss_path}", "ERROR"); sys.exit(1)
    log(f"截图尺寸: {ss_img.shape[1]}×{ss_img.shape[0]}")

    # 2. 裁剪地图
    map_img = crop_map_from_screenshot(ss_img)

    # 3. 加载缓存
    cache = load_cache(force_rebuild)
    if not cache:
        log("缓存为空", "ERROR"); sys.exit(1)

    # 4. 叠加匹配（主方法：直接比较白色环线与参考圈的吻合度）
    results = match_by_overlay(map_img, cache)

    if not results:
        log("叠加匹配失败，回退到圆检测...", "WARN")
        det = detect_ring_screenshot(map_img)
        if det is None:
            log("无法检测到缩圈。请确认截图是地图界面。", "ERROR")
            sys.exit(1)
        ring_det, _ = det
        results = match_ring(ring_det, map_img.shape, cache)

    # 从最佳匹配中获取圈信息用于可视化
    best_name = results[0][0]
    best_data = cache[best_name]
    mh, mw = map_img.shape[:2]
    md = min(mh, mw)
    ring = (best_data["cx_n"] * mw, best_data["cy_n"] * mh, best_data["r_n"] * md)

    log(f"匹配圈: center=({ring[0]:.0f},{ring[1]:.0f}) r={ring[2]:.0f}")
    log(f"归一化: cx={best_data['cx_n']:.4f} cy={best_data['cy_n']:.4f} r={best_data['r_n']:.4f}")

    if DEBUG:
        save_debug(map_img, ring, filename="detected_ring.jpg")
        save_comparison(map_img, ring, best_name, cache)

    # 5. 显示结果
    print()
    print("=" * 58)
    print(f"  {'#':<4} {'文件名':<20} {'得分':<10} {'圆心':<18} {'半径'}")
    print("-" * 58)
    for i, (name, score) in enumerate(results, 1):
        d = cache[name]
        print(f"  {i:<4} {name:<20} {score:<10.4f} "
              f"({d['cx_n']:.3f},{d['cy_n']:.3f})     {d['r_n']:.3f}")
    print("=" * 58)

    # 6. 打开
    log(f"最佳匹配: {best_name} (得分: {results[0][1]:.4f})")
    os.startfile(str((MAP_DIR / best_name).resolve()))
    log(f"已打开 {best_name}")
    log(f"耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
