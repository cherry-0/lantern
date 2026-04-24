# Noom

Closed-source weight-loss / behavior-change app. Users log meals (often via photo), body measurements, mood, and activity; the app generates personalized coaching and dietary recommendations.

## Binary

| Field | Value |
|---|---|
| Package | `com.wsl.noom` |
| Version | 14.12.0 |
| Version code | 302330 |
| Format | `.apkm` (APKMirror bundle) |
| ABI coverage | 4 architectures |
| DPI coverage | 7 |
| Locales | 4 (en + 3 others bundled) |
| Feature modules | 1 (dynamic feature split) |
| Source | APKMirror |
| File | `com.wsl.noom_14.12.0-302330_4arch_7dpi_4lang_1feat_7e7c7a76eac0cfac9835f3b156433d26_apkmirror.com.apkm` |

## Install on Verify AVD (arm64)

```bash
unzip com.wsl.noom_*.apkm -d _split/
adb install-multiple \
    _split/base.apk \
    _split/split_config.arm64_v8a.apk \
    _split/split_config.xxhdpi.apk \
    _split/split_config.en.apk \
    _split/split_<feature>.apk
```

## Expected difficulty

| Concern | Assessment |
|---|---|
| Account | **Required** |
| Subscription | **Paid only** — Noom gates ~all features behind a ~$70/month subscription after a 14-day trial. No meaningful free tier |
| TLS pinning | Medium — standard React Native + OkHttp; `apk-mitm` likely works |
| Play Integrity | Medium |
| Onboarding | Extensive (~20-screen questionnaire) before the meal-logging flow is unlocked — must be completed once and captured in the AVD snapshot |

## ⚠ Open question for user

Do you have an active Noom subscription (or a test account with one)? Without it, the meal-logging and coaching flows — the privacy-relevant ones — are not reachable. If not, recommend deferring or swapping for a free weight/meal tracker.

## Privacy relevance

**Very high.** Food photos leak diet, household composition, religious/cultural practice. Body-measurement inputs leak weight, height, BMI. Mood logs leak mental state. Geotagged photo metadata leaks location. Rich target for privacy externalization analysis, but blocked behind a paywall.

## Recommended dataset

Food-image subsets from common datasets (Food-101, Recipe1M) — not currently loaded in Verify. Alternatively, use a subset of HR-VISPR images that contain food/scene tags.

## UI flow sketch (once subscribed)

1. Launch → (skip login via AVD snapshot)
2. Bottom tab "Log" → "Meals" → "Add food" → photo picker
3. Pick `/sdcard/Download/meal.jpg` → wait for AI classification → read returned food name + calorie estimate
