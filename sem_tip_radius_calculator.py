# -*- coding: utf-8 -*-
"""SEM Tip Radius Estimation (Robust, Tuned)."""
import os
import csv
import shutil
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

try:
    import pytesseract
except Exception:
    pytesseract = None

BOTTOM_CROP_RATIO = 0.20
TOP_SEARCH_RATIO = 0.25
MIN_ROW_PIXELS = 6
WIDTH_FACTOR = 2
Y_CAP = 20
Y_START_PX = 5
GREEN_DOT_RADIUS = 0
DEBUG_SHOW = False


def load_tif_image(file_path: str) -> np.ndarray | None:
    try:
        img = Image.open(file_path).convert("L")
        print(f"이미지 로드 성공: {file_path}")
        return np.array(img)
    except Exception as exc:
        print(f"이미지 로드 중 오류 발생: {exc}")
        return None


def detect_scale_bar_pixel_length(image_bgr: np.ndarray) -> float:
    # Scale bar는 좌하단에 있으므로 ROI를 먼저 제한해 오검출을 줄입니다.
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.80):h, 0:int(w * 0.35)].copy()

    # 1) green bar detection (for colored scale bar)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 50, 40), (95, 255, 255))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    candidates = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw >= 20 and bw > bh * 2:
            candidates.append(float(bw))
    if candidates:
        return max(candidates)

    # 2) fallback: grayscale detection for white/gray/black scale bar
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad = cv2.convertScaleAbs(grad_x)
    _, edge = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(edge, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=5)
    if lines is not None:
        lengths = []
        for ln in lines[:, 0, :]:
            x1, y1, x2, y2 = ln
            if abs(y2 - y1) <= 4:  # near-horizontal
                lengths.append(float(abs(x2 - x1)))
        if lengths:
            return max(lengths)

    raise ValueError("Scale bar 픽셀 길이 자동 검출 실패 (녹색/회색 바 모두 미검출)")



def print_tesseract_install_guide() -> None:
    print("[GUIDE] Windows/Spyder에서 Tesseract 설정 방법:")
    print("  1) Tesseract OCR 설치 (예: UB Mannheim 빌드)")
    print(r"  2) 설치 경로 확인: C:\Program Files\Tesseract-OCR\tesseract.exe")
    print(r"  3) PATH에 C:\Program Files\Tesseract-OCR 추가 후 Spyder 재시작")
    print("  4) 또는 코드에서 pytesseract.pytesseract.tesseract_cmd 직접 지정")


def resolve_tesseract_cmd() -> str | None:
    if pytesseract is None:
        return None

    current = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    if current and os.path.exists(current):
        return current

    in_path = shutil.which("tesseract")
    if in_path:
        pytesseract.pytesseract.tesseract_cmd = in_path
        return in_path

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            pytesseract.pytesseract.tesseract_cmd = cand
            return cand

    return None


def detect_scale_nm_by_ocr(image_bgr: np.ndarray) -> float | None:
    if pytesseract is None:
        return None
    if resolve_tesseract_cmd() is None:
        print("[WARN] Tesseract 실행 파일을 찾지 못했습니다. OCR 대신 파일명 추정을 시도합니다.")
        print_tesseract_install_guide()
        return None

    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.80):h, 0:int(w * 0.45)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    try:
        txt = pytesseract.image_to_string(bw, config="--psm 6")
    except Exception as exc:
        print(f"[WARN] OCR 실행 실패: {exc}")
        return None

    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm", txt.lower())
    if m:
        return float(m.group(1))
    return None



def infer_scale_nm_from_filename(image_path: str) -> float | None:
    name = os.path.basename(image_path).lower()
    # Common ZEISS SEM overlays often pair these magnifications with these bars.
    mag_to_nm = {
        "200k": 50.0,
        "150k": 50.0,
        "100k": 100.0,
        "80k": 100.0,
        "50k": 200.0,
        "30k": 300.0,
        "20k": 500.0,
        "10k": 1_000.0,
    }
    for mag_key, nm in mag_to_nm.items():
        if mag_key in name:
            print(f"[INFO] 파일명 기반 scale 추정 사용: {mag_key} -> {nm:.1f} nm")
            return nm
    return None

def calibrate_scale_auto(image_bgr: np.ndarray, image_path: str | None = None) -> tuple[float | None, float]:
    # 1) scale bar의 pixel 길이 검출
    px = detect_scale_bar_pixel_length(image_bgr)
    # 2) nm 값은 OCR 우선 사용, 실패 시 파일명 fallback 사용
    nm = detect_scale_nm_by_ocr(image_bgr)
    source = "OCR"
    if nm is None:
        nm = infer_scale_nm_from_filename(image_path) if image_path is not None else None
        source = "filename"
    if nm is None:
        print(f"[WARN] nm 텍스트를 자동 인식하지 못했습니다. 픽셀 기준으로만 결과를 출력합니다. (scale bar px={px:.2f})")
        return None, px

    nm_per_pixel = nm / px
    print(f"[INFO] Scale estimated from {source}: {nm:.3f} nm / {px:.2f} px = {nm_per_pixel:.6f} nm/px")
    return nm_per_pixel, px

# (find_tip_and_edges, circle_fitting are unchanged core logic)
def find_tip_and_edges(image_gray: np.ndarray, bottom_crop_ratio: float = 0.20, top_search_ratio: float = 0.25,
                       # Tip 윤곽에서 좌/우 경계점을 추출하여 피팅용 점 집합을 만듭니다.
                       min_row_pixels: int = 6, width_factor: int = 2, y_cap: int = 20,
                       y_start_px: int = 5, debug: bool = True):
    h, _w = image_gray.shape
    y_max = int(h * (1.0 - bottom_crop_ratio))
    if y_max < int(h * 0.5):
        y_max = h
    img = image_gray[:y_max, :].copy()
    h2, _w2 = img.shape
    # 노이즈 완화 + 국부 대비 향상으로 tip 마스크 분리 성능 개선
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    proc = clahe.apply(blur)
    _, bw = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(bw) < bw.size * 0.02:
        bw = cv2.bitwise_not(bw)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cnts = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if len(cnts) == 0:
        return (None, None, {"proc": proc, "bw": bw}) if debug else (None, None)
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(bw)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)
    y_top_max = int(h2 * top_search_ratio)
    widths, rows = [], []
    for y in range(0, max(1, y_top_max)):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) < min_row_pixels:
            continue
        widths.append(int(xs.max() - xs.min() + 1)); rows.append(y)
    if len(rows) == 0:
        return (None, None, {"proc": proc, "bw": bw, "mask": mask}) if debug else (None, None)
    widths = np.array(widths); rows = np.array(rows)
    apex_y = int(rows[np.argmin(widths)])
    xs_apex = np.where(mask[apex_y, :] > 0)[0]
    apex_x = int((xs_apex.min() + xs_apex.max()) / 2)
    min_width = int(xs_apex.max() - xs_apex.min() + 1)

    # refine apex so red dot sits on the true tip end (top-most valid contour area)
    apex_x, apex_y = refine_apex_with_local_profile(image_gray=img, mask=mask, apex_x=apex_x, apex_y=apex_y)

    y0 = min(h2 - 1, apex_y + y_start_px); y1 = min(h2 - 1, apex_y + y_cap)
    left_pts, right_pts = [], []
    for y in range(y0, y1 + 1):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) < min_row_pixels:
            continue
        width = int(xs.max() - xs.min() + 1)
        if width > int(min_width * width_factor):
            break
        left_pts.append((int(xs.min()), y)); right_pts.append((int(xs.max()), y))

    # fallback: slender tips can fail strict width gating; relax locally near apex
    if len(left_pts) < 8 or len(right_pts) < 8:
        relaxed_left, relaxed_right = [], []
        y1_relaxed = min(h2 - 1, apex_y + max(y_cap * 2, 36))
        width_limit = max(int(min_width * max(width_factor * 2, 4)), min_width + 20)
        for y in range(max(0, apex_y + 2), y1_relaxed + 1):
            xs = np.where(mask[y, :] > 0)[0]
            if len(xs) < 3:
                continue
            width = int(xs.max() - xs.min() + 1)
            if width > width_limit:
                break
            relaxed_left.append((int(xs.min()), y))
            relaxed_right.append((int(xs.max()), y))
        if len(relaxed_left) >= 6 and len(relaxed_right) >= 6:
            left_pts, right_pts = relaxed_left, relaxed_right

    if len(left_pts) < 6 or len(right_pts) < 6:
        return (None, None, {"proc": proc, "bw": bw, "mask": mask, "apex": (apex_x, apex_y), "min_width": min_width}) if debug else (None, None)
    fit_points = np.array(left_pts + right_pts, dtype=np.int32)
    dbg = {"proc": proc, "bw": bw, "mask": mask, "apex": (apex_x, apex_y), "min_width": min_width, "fit_points": fit_points, "y0y1": (y0, y1)} if debug else None
    return fit_points, (apex_x, apex_y), dbg


def refine_apex_with_local_profile(image_gray: np.ndarray, mask: np.ndarray, apex_x: int, apex_y: int) -> tuple[int, int]:
    """Refine apex using the highest valid mask row and local intensity centroid.

    This reduces cases where width-min row lands a few pixels below the true tip end.
    """
    ys = np.where(mask.max(axis=1) > 0)[0]
    if ys.size == 0:
        return apex_x, apex_y

    top_y = int(ys.min())
    y1 = min(mask.shape[0] - 1, top_y + 3)
    xs = np.where(mask[top_y:y1 + 1, :] > 0)[1]
    if xs.size == 0:
        return apex_x, top_y

    x_left, x_right = int(xs.min()), int(xs.max())
    band = image_gray[top_y:y1 + 1, x_left:x_right + 1].astype(np.float32)
    if band.size == 0:
        return apex_x, top_y

    proj = band.sum(axis=0)
    if proj.sum() <= 0:
        return int((x_left + x_right) / 2), top_y

    x_coords = np.arange(x_left, x_right + 1, dtype=np.float32)
    refined_x = int(round(float((proj * x_coords).sum() / proj.sum())))
    return refined_x, top_y

def circle_fitting(points: np.ndarray, apex_point: tuple[int, int] | None = None, apex_weight: float = 5.0):
    # robust least-squares 원 피팅 (옵션: apex 통과 제약)
    if points is None or len(points) < 10:
        return None, None, None, None
    x = points[:, 0].astype(np.float64); y = points[:, 1].astype(np.float64)
    xc0, yc0 = np.mean(x), np.mean(y)
    r0 = max(np.median(np.sqrt((x - xc0) ** 2 + (y - yc0) ** 2)), 3.0)
    def residuals(p: np.ndarray) -> np.ndarray:
        xc, yc, r = p
        base = np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r
        if apex_point is None:
            return base
        ax, ay = float(apex_point[0]), float(apex_point[1])
        apex_res = np.sqrt((ax - xc) ** 2 + (ay - yc) ** 2) - r
        return np.concatenate([base, np.array([apex_weight * apex_res])])
    res = least_squares(residuals, np.array([xc0, yc0, r0]), loss="soft_l1", f_scale=2.0, max_nfev=3000)
    xc, yc, r = res.x; r = abs(float(r))
    rr = residuals(np.array([xc, yc, r])); rms = float(np.sqrt(np.mean(rr ** 2)))
    return float(xc), float(yc), r, rms

def show_result_image(image_gray: np.ndarray, tip_apex_pixel: tuple[int, int], xc: float, yc: float, r_px: float, nm_per_pixel: float | None, output_name: str, display_window: bool = False) -> None:
    display = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(display, (int(round(xc)), int(round(yc))), int(round(r_px)), (255, 0, 0), 2)
    cv2.circle(display, (int(tip_apex_pixel[0]), int(tip_apex_pixel[1])), 5, (0, 0, 255), -1)

    if nm_per_pixel is None:
        text = f"Tip Radius: {r_px:.2f} px"
    else:
        text = f"Tip Radius: {r_px * nm_per_pixel:.2f} nm"

    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    box_w = text_size[0] + 20
    box_h = text_size[1] + baseline + 20
    box_x = max(12, display.shape[1] - box_w - 12)
    box_y = 12
    cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
    cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1)
    cv2.putText(display, text, (box_x + 10, box_y + box_h - baseline - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_name, display)
    print(f"[INFO] Saved result image: {output_name}")

    if display_window:
        win = "SEM Tip Radius Result"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, display)
            cv2.waitKey(1)
            print("[INFO] 결과 창이 열렸습니다. 창을 선택한 뒤 아무 키나 누르면 닫힙니다.")
            cv2.waitKey(0)
            cv2.destroyWindow(win)
        except cv2.error as exc:
            print(f"[WARN] OpenCV 창 표시 실패: {exc}")
            print(f"[INFO] 대신 저장된 결과 파일을 확인하세요: {output_name}")


def main_tip_radius_calculator(image_path: str, output_dir: str | None = None, display_window: bool = False) -> dict:
    # 단일 이미지 처리: 로드 → 스케일 보정 → tip 검출 → 원 피팅 → 결과 저장
    image_gray = load_tif_image(image_path)
    if image_gray is None:
        return {"image": image_path, "status": "load_failed"}
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    nm_per_pixel, scale_px = calibrate_scale_auto(image_bgr, image_path)
    tip_fit_points, tip_apex_pixel, dbg = find_tip_and_edges(image_gray, BOTTOM_CROP_RATIO, TOP_SEARCH_RATIO, MIN_ROW_PIXELS, WIDTH_FACTOR, Y_CAP, Y_START_PX, True)
    if tip_fit_points is None or tip_apex_pixel is None:
        print("[ERROR] Tip apex/edge points not found. Tune parameters.")
        return {"image": image_path, "status": "tip_not_found", "scale_px": scale_px, "nm_per_pixel": nm_per_pixel}
    xc, yc, r_px, rms_px = circle_fitting(tip_fit_points, apex_point=tip_apex_pixel, apex_weight=5.0)
    if r_px is None:
        print("[ERROR] Circle fitting failed.")
        return {"image": image_path, "status": "fit_failed", "scale_px": scale_px, "nm_per_pixel": nm_per_pixel}
    print(f"[INFO] Fit points: {len(tip_fit_points)} | residual RMS: {rms_px:.3f} px")
    if nm_per_pixel is None:
        print(f"[RESULT] Tip Radius: {r_px:.2f} px (nm 변환 실패: 파일명 배율(예: 100k) 확인 필요)")
    else:
        tip_radius_nm = r_px * nm_per_pixel
        print(f"[INFO] residual RMS: {rms_px * nm_per_pixel:.3f} nm")
        print(f"[RESULT] Tip Radius: {tip_radius_nm:.2f} nm")

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = output_dir if output_dir is not None else os.path.dirname(image_path)
    output_name = os.path.join(out_dir, f"{base}_tip_radius_result.png")
    show_result_image(image_gray, tip_apex_pixel, xc, yc, r_px, nm_per_pixel, output_name, display_window=display_window)

    return {
        "image": image_path,
        "status": "ok",
        "scale_px": scale_px,
        "nm_per_pixel": nm_per_pixel,
        "radius_px": r_px,
        "radius_nm": (r_px * nm_per_pixel) if nm_per_pixel is not None else None,
        "rms_px": rms_px,
        "rms_nm": (rms_px * nm_per_pixel) if nm_per_pixel is not None else None,
        "result_png": output_name,
    }

if __name__ == "__main__":
    # 배치 실행: 폴더 내 모든 SEM 이미지를 분석하고 CSV로 요약 저장
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(title="SEM image 폴더 선택")
    if not folder:
        print("[INFO] 폴더 선택이 취소되었습니다.")
        root.destroy()
        raise SystemExit(0)

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    image_files = [
        os.path.join(folder, fn)
        for fn in sorted(os.listdir(folder))
        if os.path.splitext(fn.lower())[1] in exts
    ]

    if not image_files:
        print("[WARN] 선택한 폴더에 분석 가능한 이미지가 없습니다.")
        root.destroy()
        raise SystemExit(0)

    output_dir = os.path.join(folder, "tip_radius_results")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for idx, image_file in enumerate(image_files, start=1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")
        result = main_tip_radius_calculator(image_file, output_dir=output_dir, display_window=False)
        rows.append(result)

    csv_path = os.path.join(output_dir, "tip_radius_summary.csv")
    fieldnames = [
        "image", "status", "scale_px", "nm_per_pixel", "radius_px", "radius_nm", "rms_px", "rms_nm", "result_png"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\n[INFO] 완료: {ok_count}/{len(rows)} 성공")
    print(f"[INFO] 결과 폴더: {output_dir}")
    print(f"[INFO] CSV 저장: {csv_path}")
    root.destroy()
