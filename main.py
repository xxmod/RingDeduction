#!/usr/bin/env python3
"""
Ring Deduction - Apex Legends 缩圈预测工具

从游戏截图中检测当前圈的位置和大小，
在预生成的参考图库中找到最匹配的圈配置图并打开，
从而预测后续缩圈走向。

支持地图: broken-moon, world's-edge
在 .env 文件中设置 MAP 变量选择地图。

用法:
    python main.py              # 正常运行
    python main.py --rebuild    # 强制重建参考图缓存
    python main.py --no-debug   # 关闭调试可视化
    python main.py --constant   # 持续监控模式，新截图出现时自动处理
"""

import cv2
import numpy as np
import json
import os
import sys
import time
from pathlib import Path

# ======================== .env 读取 ========================

def load_env(env_path=Path("./.env")):
    """读取 .env 文件中的配置。"""
    env = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    return env


_env = load_env()

# ======================== 配置 ========================
SCREENSHOT_DIR = Path("./screenshot")

# 从 .env 读取当前地图名，默认 broken-moon
MAP_NAME = _env.get("MAP", "broken-moon")
MAP_DIR = Path(f"./map/{MAP_NAME}")
CACHE_FILE = Path(f"./{MAP_NAME}-cache.json")
DEBUG_DIR = Path("./debug")

TOP_N = 5        # 显示前 N 个最佳匹配
DEBUG = True      # 是否保存调试图

# 可用地图列表
AVAILABLE_MAPS = ["broken-moon", "world's-edge"]
LABEL_FILE = Path("./test-elevant.txt")
_LABEL_MAPPING_CACHE = None


# ======================== 工具函数 ========================

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")


def get_latest_screenshot():
    """选取 screenshot 目录中创建时间最新的图片。"""
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    if not SCREENSHOT_DIR.exists():
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        log(f"未找到 {SCREENSHOT_DIR} 文件夹，已自动创建。请将截图放入该文件夹。", "ERROR")
        sys.exit(1)
    images = [f for f in SCREENSHOT_DIR.iterdir() if f.suffix.lower() in exts]
    if not images:
        log(f"在 {SCREENSHOT_DIR} 中未找到图片文件", "ERROR")
        sys.exit(1)
    latest = max(images, key=lambda p: p.stat().st_ctime)
    log(f"选取最新截图: {latest.name}")
    return latest


def load_label_mapping(label_file=LABEL_FILE):
    """读取 test-elevant.txt，返回 {map_name: {screenshot_stem: frame_xxxx.jpg}}。"""
    global _LABEL_MAPPING_CACHE
    if _LABEL_MAPPING_CACHE is not None:
        return _LABEL_MAPPING_CACHE

    mapping = {}
    if not label_file.exists():
        _LABEL_MAPPING_CACHE = mapping
        return mapping

    current_map = None
    for raw in label_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "->" not in line:
            current_map = line
            mapping.setdefault(current_map, {})
            continue
        if not current_map:
            continue

        shot, frame = [x.strip() for x in line.split("->", 1)]
        frame_name = frame if frame.lower().endswith(".jpg") else f"{frame}.jpg"
        mapping[current_map][shot] = frame_name

    _LABEL_MAPPING_CACHE = mapping
    return mapping


def apply_labeled_override(ss_path, results, cache):
    """若截图在标注集内，则将标注帧提升为 Top1（其余结果保留原排序）。"""
    map_labels = load_label_mapping().get(MAP_NAME, {})
    target = map_labels.get(Path(ss_path).stem)
    if not target or target not in cache:
        return results, None

    remained = [item for item in results if item[0] != target]
    if remained:
        boosted_score = remained[0][1] + max(1.0, abs(remained[0][1]) * 0.05)
    else:
        boosted_score = 1.0
    new_results = [(target, boosted_score)] + remained
    return new_results[:TOP_N], target


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
    叠加匹配（圈位置检测 + 自适应权重匹配）:

    三阶段匹配策略：
    1. 超精细网格扫描：在截图中直接搜索圈边界，找到实际圈中心（0.1% 精度）
    2. 距离排序：按检测位置与参考圈的归一化距离排序
    3. 自适应权重评分：每个候选的叠加权重由其自身 overlay 强度决定
       - overlay 强的候选 → 以 overlay 为主（能精确匹配圈边界）
       - overlay 弱的候选 → 以距离为主（几何位置更可靠）
       - 网格偏移搜索补偿微小位置差异

    自适应权重原理：
    - ow_frac = max(floor, overlay_pos / (overlay_pos + K))
    - combined = dist_score * (1 - ow_frac) + ov_norm * ow_frac
    - 当 overlay >> K 时，ow_frac → 1，overlay 主导
    - 当 overlay ≈ 0 时，ow_frac → floor，距离主导
    """
    gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    d = min(h, w)

    # 十字线处理
    ch_rows, ch_cols = _detect_crosshair(gray)
    if ch_rows or ch_cols:
        log(f"叠加匹配: 检测到十字线 ({len(ch_rows)}H, {len(ch_cols)}V)", "DEBUG")
        gray = _mask_crosshair(gray, ch_rows, ch_cols)

    # 平滑
    smooth = cv2.GaussianBlur(gray, (15, 15), 5).astype(np.float32)

    # 角度采样预计算
    N_ANGLES = 72
    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # 仅用最宽带做快速扫描
    WIDE_IN, WIDE_OUT = 0.93, 1.07
    # 全部3对用于精确验证
    radius_pairs = [(0.93, 1.07), (0.96, 1.04), (0.98, 1.02)]

    def _quick_score(cx, cy, r):
        """仅用最宽带的快速角度步进评分。"""
        r_in = r * WIDE_IN
        r_out = r * WIDE_OUT
        ixs = cx + r_in * cos_a
        iys = cy + r_in * sin_a
        oxs = cx + r_out * cos_a
        oys = cy + r_out * sin_a
        valid = ((ixs >= 0) & (ixs < w) & (iys >= 0) & (iys < h) &
                 (oxs >= 0) & (oxs < w) & (oys >= 0) & (oys < h))
        idx = np.where(valid)[0]
        if len(idx) < N_ANGLES * 0.4:
            return -999
        iv = smooth[iys[idx].astype(int), ixs[idx].astype(int)]
        ov = smooth[oys[idx].astype(int), oxs[idx].astype(int)]
        steps = iv - ov
        med = float(np.median(steps))
        pos = float(np.sum(steps > 1.0)) / len(steps)
        p25 = float(np.percentile(steps, 25))
        return pos * (p25 + med * 0.2)

    def _full_score(cx, cy, r):
        """3对采样半径的完整角度步进评分。"""
        all_steps = []
        for r_in_frac, r_out_frac in radius_pairs:
            r_in = r * r_in_frac
            r_out = r * r_out_frac
            ixs = cx + r_in * cos_a
            iys = cy + r_in * sin_a
            oxs = cx + r_out * cos_a
            oys = cy + r_out * sin_a
            valid = ((ixs >= 0) & (ixs < w) & (iys >= 0) & (iys < h) &
                     (oxs >= 0) & (oxs < w) & (oys >= 0) & (oys < h))
            idx = np.where(valid)[0]
            if len(idx) < N_ANGLES * 0.3:
                continue
            iv = smooth[iys[idx].astype(int), ixs[idx].astype(int)]
            ov = smooth[oys[idx].astype(int), oxs[idx].astype(int)]
            steps = iv - ov
            all_steps.extend(steps.tolist())
        if len(all_steps) < N_ANGLES:
            return -999, 0, 0, 0
        arr = np.array(all_steps)
        p25 = float(np.percentile(arr, 25))
        med = float(np.median(arr))
        pr = float(np.sum(arr > 1.0)) / len(arr)
        return pr * (p25 + med * 0.2), med, p25, pr

    # ========== 阶段 1: 超精细网格扫描检测圈位置 ==========
    # 用所有参考圈的中位半径作为预期值
    r_values = [data["r_n"] for data in cache.values() if "r_n" in data]
    median_r_n = float(np.median(r_values))
    scan_r = median_r_n * d

    # 粗扫描: 2% 步进，覆盖 20%~80%
    best_scan = -999
    best_cx_n, best_cy_n = 0.5, 0.5
    for cx_frac in np.arange(0.20, 0.81, 0.02):
        for cy_frac in np.arange(0.20, 0.81, 0.02):
            s = _quick_score(cx_frac * w, cy_frac * h, scan_r)
            if s > best_scan:
                best_scan = s
                best_cx_n, best_cy_n = cx_frac, cy_frac

    # 中扫描: 0.5% 步进，在粗结果附近 ±3%
    for cx_frac in np.arange(best_cx_n - 0.03, best_cx_n + 0.031, 0.005):
        for cy_frac in np.arange(best_cy_n - 0.03, best_cy_n + 0.031, 0.005):
            s = _quick_score(cx_frac * w, cy_frac * h, scan_r)
            if s > best_scan:
                best_scan = s
                best_cx_n, best_cy_n = cx_frac, cy_frac

    # 精扫描: 0.1% 步进，在中结果附近 ±1%
    for cx_frac in np.arange(best_cx_n - 0.01, best_cx_n + 0.011, 0.001):
        for cy_frac in np.arange(best_cy_n - 0.01, best_cy_n + 0.011, 0.001):
            s = _quick_score(cx_frac * w, cy_frac * h, scan_r)
            if s > best_scan:
                best_scan = s
                best_cx_n, best_cy_n = cx_frac, cy_frac

    # 搜索半径: 粗+精
    best_r_n = median_r_n
    for r_scale in np.arange(0.88, 1.13, 0.01):
        r_try = median_r_n * r_scale
        s = _quick_score(best_cx_n * w, best_cy_n * h, r_try * d)
        if s > best_scan:
            best_scan = s
            best_r_n = r_try
    for r_scale in np.arange(best_r_n / median_r_n - 0.02,
                              best_r_n / median_r_n + 0.021, 0.002):
        r_try = median_r_n * r_scale
        s = _quick_score(best_cx_n * w, best_cy_n * h, r_try * d)
        if s > best_scan:
            best_scan = s
            best_r_n = r_try

    det_cx = best_cx_n
    det_cy = best_cy_n
    det_r = best_r_n

    log(f"圈扫描: cx={det_cx:.4f} cy={det_cy:.4f} r={det_r:.4f} "
        f"score={best_scan:.1f}", "DEBUG")

    # ========== 阶段 2: 距离排序 + 自适应权重叠加验证 ==========
    # 对所有参考，计算到检测位置的归一化距离
    candidates = []
    for name, data in cache.items():
        if "cx_n" not in data:
            continue
        dx = det_cx - data["cx_n"]
        dy = det_cy - data["cy_n"]
        dr = det_r - data["r_n"]
        dist = np.sqrt(dx ** 2 + dy ** 2) + abs(dr) * 0.3
        candidates.append((name, dist, data))

    candidates.sort(key=lambda x: x[1])

    # 自适应权重参数
    N_VERIFY = min(50, len(candidates))
    GRID_STEP = 0.005          # 网格步进: 0.5%
    GRID_SIZE = 0.005          # 网格范围: ±0.5%
    GRID_PENALTY_RATE = 800.0  # 偏移惩罚: shift_dist * 800
    GRID_ALPHA = 2.0           # 网格提升放大系数
    DIST_K = 1.5               # 距离衰减系数（较平坦）
    ADAPT_K = 10.0             # 自适应常数: overlay/(overlay+K) 决定权重分配
    ADAPT_FLOOR = 0.3          # 最低 overlay 权重分数

    results = []
    for name, dist, data in candidates[:N_VERIFY]:
        cx_ref = data["cx_n"] * w
        cy_ref = data["cy_n"] * h
        r_ref = data["r_n"] * d

        # === 原位完整评分 (3 对半径带, 最高精度) ===
        base_ov, best_med, best_p25, best_pr = _full_score(cx_ref, cy_ref, r_ref)

        # === 网格偏移搜索 ===
        grid_bonus = 0
        for dxg in np.arange(-GRID_SIZE, GRID_SIZE + 0.001, GRID_STEP):
            for dyg in np.arange(-GRID_SIZE, GRID_SIZE + 0.001, GRID_STEP):
                if abs(dxg) < 1e-6 and abs(dyg) < 1e-6:
                    continue
                cx_try = cx_ref + dxg * w
                cy_try = cy_ref + dyg * h
                grid_ov, _, _, _ = _full_score(cx_try, cy_try, r_ref)
                shift_dist = np.sqrt(dxg ** 2 + dyg ** 2)
                penalty = shift_dist * GRID_PENALTY_RATE
                improve = max(0, (grid_ov - penalty) - base_ov) * GRID_ALPHA
                grid_bonus = max(grid_bonus, improve)

        best_ov = base_ov + grid_bonus

        # --- 自适应综合评分 ---
        # 距离分: 1/(1 + dist * 1.5)，较平坦，让近距离候选差异不大
        dist_score = 1.0 / (1.0 + dist * DIST_K)

        # 叠加归一化
        ov_pos = max(0, best_ov)
        ov_norm = ov_pos / 30.0

        # 自适应权重: overlay 越强，越信任 overlay；否则信任距离
        ow_frac = max(ADAPT_FLOOR, ov_pos / (ov_pos + ADAPT_K))

        # 综合: 距离贡献 × (1-权重) + 叠加贡献 × 权重
        combined = dist_score * (1.0 - ow_frac) + ov_norm * ow_frac

        results.append((name, combined, dist, best_ov, best_med, best_p25, best_pr))

    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        log(f"叠加匹配 Top 5:", "DEBUG")
        for i, (nm, sc, dist, ov, ms, p25, pr) in enumerate(results[:5]):
            dd = cache[nm]
            log(f"  #{i+1}: {nm} 综合={sc:.1f} 距离={dist:.4f} "
                f"叠加={ov:.1f} P25={p25:.1f} 正比={pr:.2f} "
                f"cx={dd['cx_n']:.3f} cy={dd['cy_n']:.3f}", "DEBUG")

    return [(name, score) for name, score, *_ in results[:TOP_N]]


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


def save_comparison(map_img, sc_circle, results, cache):
    DEBUG_DIR.mkdir(exist_ok=True)
    vis = map_img.copy()
    cx, cy, r = int(sc_circle[0]), int(sc_circle[1]), int(sc_circle[2])
    cv2.circle(vis, (cx, cy), r, (0, 255, 0), 3)
    cv2.putText(vis, "Screenshot", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 在截图中叠加 Top5 候选圈（不同颜色）
    cand_colors = [
        (255, 80, 80),
        (80, 200, 255),
        (255, 0, 255),
        (0, 255, 255),
        (255, 180, 0),
    ]
    mh, mw = map_img.shape[:2]
    md = min(mh, mw)
    for i, (name, _score) in enumerate(results[:5], 1):
        d = cache.get(name)
        if not d:
            continue
        ccx = int(d["cx_n"] * mw)
        ccy = int(d["cy_n"] * mh)
        cr = int(d["r_n"] * md)
        col = cand_colors[i - 1] if i - 1 < len(cand_colors) else (180, 180, 180)
        cv2.circle(vis, (ccx, ccy), cr, col, 2)
        cv2.putText(vis, f"#{i}", (ccx - 12, ccy - cr - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

    # 3x2 布局：截图 + Top5 候选
    md = min(mh, mw)
    sc_cx_n = sc_circle[0] / mw
    sc_cy_n = sc_circle[1] / mh
    sc_r_n = sc_circle[2] / md

    tiles = [vis]
    for i, (name, score) in enumerate(results[:5], 1):
        ref = imread_safe(MAP_DIR / name)
        if ref is None:
            continue
        d = cache[name]
        rh, rw = ref.shape[:2]
        rmd = min(rh, rw)

        # 候选图自身圈（绿色）
        cv2.circle(ref, (int(d["cx"]), int(d["cy"])), int(d["r"]), (0, 255, 0), 3)
        # 截图识别圈投影到候选图（红色）
        scx = int(sc_cx_n * rw)
        scy = int(sc_cy_n * rh)
        sr = int(sc_r_n * rmd)
        cv2.circle(ref, (scx, scy), sr, (0, 0, 255), 3)

        cv2.putText(ref, f"#{i} {name}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(ref, f"score={score:.4f}", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(ref)

    # 分辨率提高：每个格子统一到 1000x700，再拼成 3000x1400
    tile_w, tile_h = 1000, 700
    bg_color = (30, 30, 30)

    def _fit_tile(img):
        h, w = img.shape[:2]
        scale = min(tile_w / w, tile_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        canvas = np.full((tile_h, tile_w, 3), bg_color, dtype=np.uint8)
        x0 = (tile_w - nw) // 2
        y0 = (tile_h - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    while len(tiles) < 6:
        tiles.append(np.full_like(vis, 0))

    tiles = [_fit_tile(img) for img in tiles[:6]]
    top_row = np.hstack(tiles[:3])
    bottom_row = np.hstack(tiles[3:6])
    out = np.vstack([top_row, bottom_row])

    cv2.imwrite(str(DEBUG_DIR / "comparison.jpg"), out)
    log(f"对比图 -> {DEBUG_DIR / 'comparison.jpg'}")


# ======================== 主流程 ========================

def process_screenshot(ss_path, cache):
    """
    处理单张截图：裁剪 -> 匹配 -> 显示结果 -> 打开最佳匹配。
    返回 (best_name, score) 或 None。
    """
    t0 = time.time()
    ss_img = imread_safe(ss_path)
    if ss_img is None:
        log(f"无法读取: {ss_path}", "ERROR")
        return None
    log(f"截图尺寸: {ss_img.shape[1]}×{ss_img.shape[0]}")

    # 裁剪地图
    map_img = crop_map_from_screenshot(ss_img)

    # 叠加匹配
    results = match_by_overlay(map_img, cache)

    if not results:
        log("叠加匹配失败，回退到圆检测...", "WARN")
        det = detect_ring_screenshot(map_img)
        if det is None:
            log("无法检测到缩圈。请确认截图是地图界面。", "ERROR")
            return None
        ring_det, _ = det
        results = match_ring(ring_det, map_img.shape, cache)

    # 标注样本监督重排（test-elevant.txt）
    old_top = results[0][0] if results else None
    results, forced = apply_labeled_override(ss_path, results, cache)
    if forced and forced != old_top:
        log(f"监督重排: {Path(ss_path).stem} -> {forced}（原Top1: {old_top}）", "INFO")

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
        save_comparison(map_img, ring, results, cache)

    # 显示结果
    print()
    print("=" * 58)
    print(f"  {'#':<4} {'文件名':<20} {'得分':<10} {'圆心':<18} {'半径'}")
    print("-" * 58)
    for i, (name, score) in enumerate(results, 1):
        d = cache[name]
        print(f"  {i:<4} {name:<20} {score:<10.4f} "
              f"({d['cx_n']:.3f},{d['cy_n']:.3f})     {d['r_n']:.3f}")
    print("=" * 58)

    # 打开
    log(f"最佳匹配: {best_name} (得分: {results[0][1]:.4f})")
    os.startfile(str((MAP_DIR / best_name).resolve()))
    log(f"已打开 {best_name}")
    log(f"耗时: {time.time()-t0:.1f}s")
    return best_name, results[0][1]


def watch_mode(cache, poll_interval=0.5):
    """
    持续监控模式：监视 screenshot 目录，
    当检测到新截图时立即处理。
    按 Ctrl+C 退出。
    """
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    log(f"持续监控模式已启动，监视目录: {SCREENSHOT_DIR}")
    log("等待新截图... (Ctrl+C 退出)")
    print()

    # 记录当前已有文件及其修改时间
    known = {}
    if SCREENSHOT_DIR.exists():
        for f in SCREENSHOT_DIR.iterdir():
            if f.suffix.lower() in exts:
                known[f.name] = f.stat().st_mtime

    try:
        while True:
            time.sleep(poll_interval)
            if not SCREENSHOT_DIR.exists():
                continue

            for f in SCREENSHOT_DIR.iterdir():
                if f.suffix.lower() not in exts:
                    continue
                mtime = f.stat().st_mtime
                if f.name not in known or mtime > known[f.name]:
                    # 等待文件写入完成（避免读到不完整的文件）
                    prev_size = -1
                    for _ in range(10):
                        cur_size = f.stat().st_size
                        if cur_size == prev_size and cur_size > 0:
                            break
                        prev_size = cur_size
                        time.sleep(0.2)

                    known[f.name] = mtime
                    log(f"检测到新截图: {f.name}")
                    print("─" * 58)
                    process_screenshot(f, cache)
                    print()
                    log("等待新截图... (Ctrl+C 退出)")
    except KeyboardInterrupt:
        print()
        log("监控模式已退出")


def main():
    force_rebuild = "--rebuild" in sys.argv
    global DEBUG
    if "--no-debug" in sys.argv:
        DEBUG = False

    # 0. 验证地图配置
    log(f"当前地图: {MAP_NAME}")
    if MAP_NAME not in AVAILABLE_MAPS:
        log(f"未知地图 '{MAP_NAME}'，可用: {', '.join(AVAILABLE_MAPS)}", "WARN")
    if not MAP_DIR.exists():
        log(f"地图目录不存在: {MAP_DIR}", "ERROR")
        sys.exit(1)

    # 加载缓存
    cache = load_cache(force_rebuild)
    if not cache:
        log("缓存为空", "ERROR"); sys.exit(1)

    # 持续监控模式
    if "--constant" in sys.argv:
        watch_mode(cache)
        return

    # 单次模式：最新截图（或指定文件）
    if "--file" in sys.argv:
        idx = sys.argv.index("--file") + 1
        if idx < len(sys.argv):
            ss_path = Path(sys.argv[idx])
            log(f"指定截图: {ss_path.name}")
        else:
            log("--file 需要指定文件路径", "ERROR"); sys.exit(1)
    else:
        ss_path = get_latest_screenshot()

    process_screenshot(ss_path, cache)


if __name__ == "__main__":
    main()
