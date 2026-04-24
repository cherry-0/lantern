# Photomath

Closed-source math problem solver. Points the camera at a handwritten or printed equation; the app OCRs the expression and returns a step-by-step solution.

## Binary

| Field | Value |
|---|---|
| Package | `com.microblink.photomath` |
| Version | 8.47.1 |
| Version code | 70001015 |
| Format | `.apkm` (APKMirror bundle) |
| ABI coverage | 4 architectures (`armeabi-v7a`, `arm64-v8a`, `x86`, `x86_64`) |
| DPI coverage | 7 (nodpi, ldpi…xxxhdpi) |
| Source | APKMirror |
| File | `com.microblink.photomath_8.47.1-70001015_4arch_7dpi_f3942b2ec0cd28e15de4229bfd459de2_apkmirror.com.apkm` |

## Install on Verify AVD (arm64)

```bash
unzip com.microblink.photomath_*.apkm -d _split/
adb install-multiple \
    _split/base.apk \
    _split/split_config.arm64_v8a.apk \
    _split/split_config.xxhdpi.apk \
    _split/split_config.en.apk
```

The helper `verify/backend/drivers/_apkm_installer.py` automates this selection based on the AVD's `ro.product.cpu.abi` and `wm density`.

## Expected difficulty

| Concern | Assessment |
|---|---|
| Google account login | Optional on the free tier (supports skip-sign-in) |
| Subscription | Basic OCR + step-by-step is free; "Photomath Plus" explanations require paid tier |
| TLS pinning | Moderate — owned by Google since 2022, uses OkHttp; `apk-mitm` expected to work |
| Play Integrity | Low-medium — not a payment app, unlikely to hard-fail on AVD |
| Camera requirement | **Yes** — primary input is camera. Must push image to `/sdcard/DCIM/` and use the "gallery import" flow instead of live camera |

## Privacy relevance

Low-to-medium. Input images are math problems (handwriting style may leak identity in principle), which is the weakest privacy story of the five closed-source apps. Useful primarily as a **tractability benchmark** for the black-box pipeline — not as a high-signal privacy test.

## Recommended dataset

Synthetic: render math expressions on plain backgrounds. No existing Verify dataset fits directly; may need a small custom image set.
