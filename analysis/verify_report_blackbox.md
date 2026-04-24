# Verify Framework — Black-Box (Closed-Source) App Extension

This document describes how to extend the `verify/` privacy evaluation framework from open-source target apps to **closed-source Android apps driven inside an emulator**. It follows the structure and conventions of [`verify_report.md`](./verify_report.md); refer to that document for the current (open-source) architecture.

---

## 1. Why the Current Architecture Does Not Apply

The existing pipeline has three load-bearing assumptions that all require source access:

| Assumption (open-source) | Why it fails for closed-source |
|---|---|
| Runner imports app internals directly (`from apps.ai.services.chains import subject_chain`) | Closed-source apps expose no Python entry point — only an installed APK |
| `_runtime_capture` monkey-patches `urllib3` / `httpx` / `builtins.open` / Django signals *inside the app process* | We do not control the app process; it runs in the Android Runtime, not a Python interpreter |
| `set_phase("POST")` is inserted at the exact line where inference ends | No source to modify; phase boundary must be inferred from external observation |

What **does** port unchanged:

- Dataset loading (`verify/backend/datasets/`)
- Attribute filtering (`label_mapper.py`)
- Perturbation methods (`verify/backend/perturbation_method/*`)
- Inferability evaluation (`verify/backend/evaluation_method/evaluator.py`)
- Orchestrator loop and cache layer (`verify/backend/orchestrator.py`, `cache.py`)
- `AdapterResult` contract and `combined_output` evaluation surface

All new work is concentrated in **adapters, runners, and externalization capture** — the boundary between the framework and the target app.

---

## 2. Target Scope: Android Emulator

This extension commits to a single execution environment: the **Android Emulator (AVD)** shipped with the Android SDK. Rationale:

- **Deterministic and snapshottable.** AVD snapshots let every pipeline run start from an identical device state (logged-in account, consented dialogs, cleared caches). No real-device state drift.
- **Rootable.** Google-provided AOSP images are rootable out-of-the-box; Play Store images are not, but can be swapped when pinning bypass is required.
- **Headless-capable.** `emulator -no-window -no-audio -no-boot-anim` runs on a server.
- **No physical hardware dependency.** The whole `verify/` run stays on one developer machine.

Other closed-source shapes (hosted APIs, desktop binaries, iOS apps) are explicitly **out of scope** for this document; they would require parallel driver/observer implementations and are tracked as future work (§10).

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Orchestrator (unchanged)                    │
│                                                                     │
│   for item in dataset:                                              │
│       adapter.run_pipeline(item)                                    │
│       perturb(item)                                                 │
│       adapter.run_pipeline(perturbed_item)                          │
│       evaluate_both(orig, pert, attributes)                         │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               BlackBoxAdapter  (new — inherits BaseAdapter)         │
│                                                                     │
│   run_pipeline(item):                                               │
│       1. ensure device ready (boot emulator, restore snapshot)      │
│       2. start observers                                            │
│       3. driver: launch app, input item, submit, wait               │
│       4. stop observers                                             │
│       5. phase_classifier: split events into DURING / POST          │
│       6. return AdapterResult(externalizations = POST events)       │
└─────┬──────────────────────────┬────────────────────────────────────┘
      │                          │
      ▼                          ▼
┌──────────────┐       ┌─────────────────────────────────────────────┐
│ AndroidDriver│       │             Observers (parallel)            │
│              │       │                                             │
│ uiautomator2 │       │  NetworkObserver  — mitmproxy flows         │
│  (Python)    │       │  FsObserver       — adb + /data/data diff   │
│              │       │  LogObserver      — adb logcat              │
│ adb shell …  │       │  UiObserver       — uiautomator hierarchy   │
└──────────────┘       └─────────────────────────────────────────────┘
```

**What's new**, in `verify/backend/`:

- `adapters/blackbox_base.py` — `BlackBoxAdapter` base class
- `adapters/<appname>_blackbox.py` — per-app subclasses with locator config
- `observers/` (new package) — `network_observer.py`, `fs_observer.py`, `log_observer.py`, `ui_observer.py`, `phase_classifier.py`
- `drivers/` (new package) — `emulator_manager.py`, `android_driver.py`
- `runners/` is **unused** for black-box adapters (no subprocess, no conda env)

---

## 4. New Component: `BlackBoxAdapter`

Lives at `verify/backend/adapters/blackbox_base.py`. Inherits `BaseAdapter` so the orchestrator cannot tell the difference.

```python
from verify.backend.adapters.base import BaseAdapter, AdapterResult
from verify.backend.drivers.emulator_manager import EmulatorManager
from verify.backend.drivers.android_driver import AndroidDriver
from verify.backend.observers import (
    NetworkObserver, FsObserver, LogObserver, UiObserver, classify_phases,
)


class BlackBoxAdapter(BaseAdapter):
    # Per-subclass configuration (set by concrete adapters)
    apk_path: str                    # path to the APK (installed once per env)
    package_name: str                # e.g. "com.example.photoai"
    main_activity: str               # e.g. ".MainActivity"
    avd_name: str = "verify_pixel7"
    snapshot_name: str = "clean"     # snapshot to restore before each run
    llm_hosts: list[str] = []        # hostnames treated as inference providers
    pinning_bypass: str = "none"     # "none" | "apkmitm" | "frida"

    # Subclass-provided UI flow
    def _drive_app(self, driver: AndroidDriver, item: dict) -> str:
        """Input the dataset item into the UI; return the app's response text."""
        raise NotImplementedError

    def check_availability(self):
        return EmulatorManager.probe(self.avd_name)

    def run_pipeline(self, input_item):
        em = EmulatorManager(self.avd_name)
        em.ensure_booted()
        em.restore_snapshot(self.snapshot_name)

        net = NetworkObserver(proxy_port=em.proxy_port)
        fs  = FsObserver(em.serial, self.package_name)
        log = LogObserver(em.serial, self.package_name)
        ui  = UiObserver(em.serial)

        with net, fs, log, ui:
            driver = AndroidDriver(em.serial, self.package_name, self.main_activity)
            driver.launch()
            response_text = self._drive_app(driver, input_item)
            driver.stop()

        events = {
            "NETWORK": net.flows,
            "STORAGE": fs.diff,
            "LOGGING": log.records,
            "UI":      ui.events,
        }
        during, post = classify_phases(events, llm_hosts=self.llm_hosts)

        return AdapterResult(
            success=True,
            output_text=response_text,
            externalizations=self._flatten(post),
            raw_output={"during": during, "post": post},
            metadata={"method": "blackbox_android"},
        )
```

Every concrete adapter only provides:

1. **Constants:** `apk_path`, `package_name`, `main_activity`, `llm_hosts`.
2. **`_drive_app(driver, item)`** — the UI flow (tap here, type this, read that).
3. **Optional:** `check_availability()` overrides, a serverless fallback (usually skipped — see §8).

This is dramatically smaller than the open-source adapter (no runner, no conda env, no settings shim).

---

## 5. New Component: Emulator Manager

`verify/backend/drivers/emulator_manager.py` wraps the Android SDK CLIs. Responsibilities:

| Method | What it runs | Purpose |
|---|---|---|
| `probe(avd)` | `emulator -list-avds` | Non-blocking availability check |
| `ensure_booted()` | `emulator -avd <name> -no-window -no-audio -http-proxy 127.0.0.1:<port>` + `adb wait-for-device` + poll `getprop sys.boot_completed` | Boot the AVD if not already running |
| `install_apk(path)` | `adb install -r <path>` | One-time per AVD |
| `restore_snapshot(name)` | `adb emu avd snapshot load <name>` | Reset state between runs |
| `set_proxy(port)` | `adb shell settings put global http_proxy 10.0.2.2:<port>` | Route traffic through mitmproxy |
| `install_mitm_ca()` | `adb push mitmproxy-ca-cert.pem /sdcard/` + settings import (rooted AVD only) | Trust our interception CA |
| `grant_permissions(perms)` | `adb shell pm grant <pkg> <perm>` | Skip runtime permission dialogs |
| `shutdown()` | `adb emu kill` | Clean teardown |

**Snapshot strategy.** The intended workflow is one-time manual provisioning:

1. Create AVD (`avdmanager create avd -n verify_pixel7 -k "system-images;android-34;google_apis;arm64-v8a"`).
2. Boot with headed UI, manually: click through first-run wizard, sign into any test Google account, grant Play Services consent.
3. Install every target APK with `install_apk`.
4. For each target, launch once, dismiss onboarding / cookie / login dialogs, log in with a test account, accept permissions.
5. Save AVD snapshot: `adb emu avd snapshot save clean`.

From then on, every `verify/` run calls `restore_snapshot("clean")` and gets an identical, pre-logged-in device. This replaces the per-app conda env setup from the open-source path and has comparable one-time cost.

---

## 6. Observers (the `_runtime_capture` replacement)

All observers implement `start()` / `stop()` and produce a `events` list with `(timestamp, channel, payload)` tuples. They run **out-of-process** and therefore do not require any access to the app's code.

### 6.1 `NetworkObserver` — mitmproxy

- Spawns `mitmdump -p <port> -s verify/backend/observers/_mitm_addon.py --set flow_detail=0`.
- The addon writes each flow as a JSON line to a temp file.
- On `stop()`, the observer reads the file and parses into events:
  - `method`, `scheme://host:port/path`, `status`, `request_bytes`, `response_bytes`, `timestamp`
  - Response body retained for known JSON content-types; truncated at 4 KB.
- **CA trust:** `mitmproxy-ca-cert.pem` from `~/.mitmproxy/` is pushed to `/system/etc/security/cacerts/` on a rooted AVD at snapshot-provisioning time. After this, the emulator trusts mitmproxy's CA system-wide.
- **User-installed CA on Android 7+:** apps that target API 24+ ignore user CAs by default unless they opt in via `networkSecurityConfig`. Hence the system-CA install above is the robust path.

### 6.2 `FsObserver` — adb snapshot diff

Android emulator has no `fs_usage` equivalent. Use before/after directory hashing:

1. Before app launch: `adb shell "find /data/data/<pkg> /sdcard -type f -exec stat -c '%n %Y %s' {} \;"` → serialized to JSON.
2. After app run: same listing.
3. Diff → STORAGE events: new files, modified files (size or mtime change).

Filter out noise paths: `cache/`, `code_cache/`, `*.lock`, `*.tmp`. Keep interesting extensions (`.json`, `.db`, `.sqlite*`, `.jpg`, `.png`, `.pdf`, `.txt`, `.csv`).

Requires rooted AVD for `/data/data/<pkg>`; non-root AVDs can still observe `/sdcard`.

### 6.3 `LogObserver` — logcat

`adb logcat -s <tag>:V` in a background `Popen`. Filter by package tag (or UID if the app logs under system tags). `stop()` terminates the subprocess and parses lines. Cap at 200 records, then prioritize WARNING+ (matches `_runtime_capture.py`'s logging filter).

### 6.4 `UiObserver` — uiautomator hierarchy snapshots

`uiautomator2`'s `d.dump_hierarchy()` returns the accessibility XML. Sampled:

- Once immediately after app launch (baseline UI).
- Once after the response is rendered (what is shown to the user).

The diff between baseline and post-response is converted to UI events: newly visible text nodes, new image-resource nodes, new notification intents. This captures "what the app *displayed* about the private data" — the black-box analogue of the open-source UI channel.

### 6.5 Phase Classification

`verify/backend/observers/phase_classifier.py` replaces `set_phase("POST")`. Heuristic:

1. Scan NETWORK events for requests whose host matches a known inference-provider pattern: `*.openai.com`, `*.anthropic.com`, `generativelanguage.googleapis.com`, `api.together.xyz`, `openrouter.ai`, plus app-specific patterns declared on the adapter.
2. The **last response** from any matched host is the inference boundary `t*`.
3. All events with `timestamp < t*` → DURING (discarded, same as open-source).
4. All events with `timestamp >= t*` → POST (returned as externalizations).

**Fallback** when no LLM host is matched (e.g. app hits a custom backend that proxies inference internally): treat the **first response** from the app's primary backend domain as `t*`. The adapter declares `primary_backend_host` as a config field.

**Known limitation.** If the app does multi-turn inference (N LLM calls), we take the last one. Adapters with known multi-call patterns can override `classify_phases` to pick a different boundary.

---

## 7. New Component: `AndroidDriver`

`verify/backend/drivers/android_driver.py` wraps `uiautomator2`:

```python
import uiautomator2 as u2

class AndroidDriver:
    def __init__(self, serial, package, activity):
        self.d = u2.connect(serial)
        self.package, self.activity = package, activity

    def launch(self):
        self.d.app_start(self.package, self.activity, stop=True)

    def type_into(self, locator, text):
        self.d(**locator).set_text(text)

    def tap(self, locator):
        self.d(**locator).click()

    def read(self, locator, timeout=30):
        return self.d(**locator).get_text(timeout=timeout)

    def wait_for(self, locator, timeout=30):
        return self.d(**locator).wait(timeout=timeout)

    def screenshot(self, path):
        self.d.screenshot(path)

    def stop(self):
        self.d.app_stop(self.package)
```

Locators are plain dicts: `{"resourceId": "com.app:id/input"}`, `{"text": "Send"}`, `{"className": "android.widget.EditText", "instance": 0}`.

Per-adapter UI flow example (`adapters/photoai_blackbox.py`):

```python
class PhotoAIAdapter(BlackBoxAdapter):
    package_name = "com.example.photoai"
    main_activity = ".MainActivity"
    llm_hosts = ["*.openai.com"]
    apk_path  = "vendor-apks/photoai-3.2.1.apk"

    def _drive_app(self, driver, item):
        # dataset item is an image
        push_image_to_emulator(item["data"], "/sdcard/Download/input.jpg")
        driver.tap({"resourceId": "com.example.photoai:id/upload_btn"})
        driver.tap({"text": "input.jpg"})
        driver.tap({"resourceId": "com.example.photoai:id/analyze"})
        return driver.read({"resourceId": "com.example.photoai:id/result_text"}, timeout=60)
```

---

## 8. Serverless Fallback for Black-Box Adapters

The open-source fallback (`USE_APP_SERVERS=false`) works because we know the app's *internal* prompt and can replicate it via OpenRouter. For closed-source apps we do not know the prompt.

**Decision: drop serverless for black-box adapters in v1.** `check_availability()` returns `(False, "[BLACKBOX] Requires USE_APP_SERVERS=true + Android emulator.")` when `use_app_servers()` is false. The adapter is silently skipped in serverless mode rather than producing misleading fallback data.

A future enhancement (tracked in §10) is to capture one representative NETWORK flow during an initial native run, derive a **functional spec** (outbound request template + output shape), and use that to construct a serverless mimic. This is non-trivial and deferred.

---

## 9. Certificate Pinning and Emulator Detection

Commercial apps frequently defend against MITM. Handling matrix:

| Defense | What fails | Mitigation | Cost |
|---|---|---|---|
| System CA not trusted by app (opt-out via `networkSecurityConfig`) | mitmproxy sees TLS handshake but app rejects our cert | Rewrite `AndroidManifest.xml` with `apk-mitm` to allow user CAs | ~30 s per APK, one-time |
| Public-key pinning (`OkHttp` CertificatePinner, `TrustManager` overrides) | Even system CA installed, app hard-fails on unknown public key | `apk-mitm` strips pinners in most cases; `frida` + `objection --codeshare pinning-bypass` otherwise | `apk-mitm`: automatic; `frida`: needs rooted AVD |
| Root/emulator detection (checks `Build.FINGERPRINT`, `/system/bin/su`, Play Integrity) | App refuses to start on AVD | Use Google-API AOSP image (not Play Store image), MagiskHide + Shamiko on rooted AVD | High — per-app cat-and-mouse |
| SafetyNet / Play Integrity hardware attestation | Full refusal — banking grade | Out of scope; skip app | — |

**Recommended default stack per target:**

1. Try the vanilla APK on the AVD first; about half of consumer AI apps work as-is with a system-installed mitmproxy CA.
2. If TLS intercepts fail, run `apk-mitm app.apk` → install the patched APK. No root required.
3. If the app still refuses (hard pinning), switch to a rooted AVD + Frida pinning bypass.
4. If Play Integrity blocks, mark the app unsupported and move on.

Per-adapter `pinning_bypass` config selects which path is applied at snapshot-provisioning time.

---

## 10. Comparison: Open-Source vs. Black-Box

| Concern | Open-source (current) | Black-box (this doc) |
|---|---|---|
| Install target | `conda create` + `pip install -r` | `adb install app.apk` + AVD snapshot |
| Invoke target | Runner subprocess imports app modules | `uiautomator2` taps in the UI |
| Phase boundary | `_runtime_capture.set_phase("POST")` inserted in runner | Timestamp split at last LLM-host response |
| NETWORK capture | `urllib3`/`httpx` monkey-patch in runner | `mitmproxy` with system-CA on rooted AVD |
| STORAGE capture | `post_save` signal + `builtins.open` patch | Before/after `/data/data/<pkg>` diff |
| LOGGING capture | Root logger handler in runner | `adb logcat` background process |
| UI capture | Patched Django Channels / Starlette WS | `uiautomator2` hierarchy diff |
| IPC capture | Redis / socket / Popen patches | Not implemented (skipped in v1) |
| Serverless fallback | OpenRouter replicates app prompt | Not supported in v1 |
| Per-app setup cost | ~1 conda env + ~100-line runner + optional shim | Locator config + APK snapshot (~2-4 hours) |
| Maintenance burden | Dependency drift when app updates `requirements.txt` | Locator drift when app updates UI |
| Legal posture | We have source; no ToS tension | MITM of third-party apps is research-use-only; document per adapter |

---

## 11. Implementation Plan & Progress

This section is the record of what was originally planned versus what actually landed, and why the plan changed where it did. §11.1 is the spec as first written (preserved verbatim). §11.2 tracks delivery per phase. §11.3 is the explicit list of plan deltas with rationale.

### 11.1 Original plan (as first specified)

#### Phase 1 — Scaffolding

- Add `verify/backend/observers/` and `verify/backend/drivers/` packages.
- Implement `EmulatorManager` with `probe`, `ensure_booted`, `shutdown`, `restore_snapshot`, `set_proxy`.
- Implement `NetworkObserver` (mitmdump + JSON lines addon).
- Implement `phase_classifier.classify_phases` with a hardcoded list of common LLM hosts.
- Add `BlackBoxAdapter` base class in `adapters/blackbox_base.py`.
- Smoke test: a dummy adapter that launches Chrome on the emulator, navigates to a public LLM chat URL, captures the outbound network. This validates the observer/driver contract without any target-app complexity.

#### Phase 2 — First real target

- Pick one consumer AI app with a simple UI flow and **no aggressive pinning** (e.g. a receipt scanner, a photo caption app). Verify the app runs on the AVD unmodified.
- Provision the `verify_pixel7` AVD: boot, install APK, run through onboarding, save `clean` snapshot.
- Write `adapters/<name>_blackbox.py` with `_drive_app` and locator config.
- Add `FsObserver`, `LogObserver`, `UiObserver`.
- Run one dataset item end-to-end through orchestrator. Confirm externalizations appear in Streamlit.

#### Phase 3 — Hardening

- Add `apk-mitm` wrapper for pinning bypass.
- Rooted AVD + Frida path for apps that still resist.
- Per-adapter `primary_backend_host` fallback in phase classifier.
- Frontend: add a "device required" badge and emulator-not-running status indicator.

#### Phase 4 — Expansion

- Port 2-3 additional target apps to validate the locator-maintenance cost model.
- Extend [`verify_report.md`](./verify_report.md) §7 with a black-box variant of the adapter template.
- Extend [`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md) with emulator / mitmproxy / pinning sections.

#### Phase 5 — Deferred

- Serverless fallback via captured functional spec (§8).
- iOS support (simulator is sandboxed differently; would need parallel `IosDriver` + `XCUITest` work).
- Hosted-API-only targets (pure `ApiDriver`, no emulator).
- IPC channel (Binder via `strace` or `sysdig` on the AVD host).

### 11.2 Delivered vs. planned

Legend: ✅ delivered · ⏩ pulled forward from a later phase · ⏸ deferred · ❌ skipped (with rationale).

#### Phase 1 — Scaffolding (complete)

| Item (as specified) | Status | Notes |
|---|---|---|
| `observers/` + `drivers/` packages | ✅ | `verify/backend/observers/`, `verify/backend/drivers/`, each with `__init__.py`. |
| `EmulatorManager` with probe/boot/shutdown/restore/set_proxy | ✅ (see §11.3(a) for the proxy-handling reversal) | After an initial attempt at a boot-time `-http-proxy`, reverted to runtime toggling via `set_runtime_proxy` / `clear_runtime_proxy` so the AVD has normal internet during provisioning/onboarding and only routes through mitmdump while `NetworkObserver` is active. Gained `install_apk`, `install_apkm`, `save_snapshot`, `grant_permissions`, `push_file`, `get_prop`, `get_density`. |
| `NetworkObserver` + mitmdump addon | ✅ | `_mitm_addon.py` emits JSON-lines per flow with 4 KB body cap for JSON content-types; observer parses on stop. |
| `phase_classifier.classify_phases` with hardcoded LLM hosts | ✅ with scope creep | Ships 14 fnmatch patterns + an exclude list of non-inference `*.googleapis.com` endpoints (Firebase, Play, crash reporting). Also accepts `primary_backend_host` fallback — originally a Phase 3 item. See §11.3(b). |
| `BlackBoxAdapter` base class | ✅ | `adapters/blackbox_base.py` with `BlackBoxConfig` dataclass so subclasses only override `_drive_app` + class-level config. |
| Chrome-on-emulator smoke test | ❌ replaced | Never run — no SDK on dev machine at scaffolding time. Replaced by `verify/backend/tests/test_blackbox_observers.py` (9 synthetic-data pytest cases covering UiObserver diff, LogObserver parsing/truncation, phase classification across 4 channels). See §11.3(c). |

#### Phase 2 — First real target (code complete; device validation pending)

| Item (as specified) | Status | Notes |
|---|---|---|
| Pick easy target | ✅ | Photomath 8.47.1 chosen (low pinning, low privacy stakes, offered as tractability benchmark). |
| Provision AVD (boot + install + onboarding + snapshot) | ⏸ user-blocked | Manual click-through cannot be automated across vendors; replaced with `drivers/provision.py` which automates the non-interactive parts (boot, mitmproxy CA install, APK(M) install, `save_snapshot`) around an explicit user pause. See §11.3(d). |
| `_drive_app` + locator config for Photomath | ✅ provisional | `adapters/photomath_blackbox.py` registered as `photomath`. Locators are fnmatch-style best-effort guesses; §15.5 walks the user through verifying them against a real `dump_hierarchy`. |
| `FsObserver`, `LogObserver`, `UiObserver` | ✅ | All three land in `observers/`; `BlackBoxAdapter.run_pipeline` now enters all four observers in one `with` stack; `_flatten_post` formats STORAGE/LOGGING/UI events channel-appropriately. |
| End-to-end dataset item through orchestrator | ⏸ user-blocked | Waits on §15.7 (first real run after AVD is provisioned). |

#### Phase 3 — Hardening (partial, two items pulled forward)

| Item (as specified) | Status | Notes |
|---|---|---|
| `apk-mitm` wrapper for pinning bypass | ⏸ | Not started. Photomath's moderate pinning expected to pass with system CA alone; Replika/Expensify will force this work later. |
| Rooted AVD + Frida path | ⏸ | Not started. |
| Per-adapter `primary_backend_host` fallback | ⏩ landed in Phase 1 | Lives in `phase_classifier.py` + `BlackBoxConfig.primary_backend_host`. Photomath uses `*.photomath.net`. |
| Frontend "device required" badge | ⏩ landed in Phase 2 | `0_Initialization.py` now branches on `isinstance(adapter, BlackBoxAdapter)` and renders `"black-box · device required"` in place of the Python version. `KNOWN_APPS` in pages 1 and 2 extended. |
| Emulator-not-running status indicator | ⏸ partial | `check_availability()` already surfaces "emulator not on PATH" via `EmulatorManager.probe`; a dedicated badge beyond that is not yet wired. |

#### Phase 4 — Expansion (partial)

| Item (as specified) | Status | Notes |
|---|---|---|
| 2–3 additional targets | ⏩ partial (stubs) | `adapters/replika_blackbox.py` (`replika`, text, `*.replika.ai`) and `adapters/expensify_blackbox.py` (`expensify`, image, `*.expensify.com`) ship with populated config and registered classes; `_drive_app` raises `NotImplementedError` until locators are verified. Pulled forward so the registry/config surface is final before device time is spent. See §11.3(e). |
| `verify_report.md` §7 black-box variant | ⏸ | Not started. |
| `TROUBLESHOOTING.md` extension | ⏸ | Not started (existing §15.8 table in this doc is a provisional substitute). |

#### Phase 5 — Deferred (unchanged)

Serverless fallback, iOS, hosted-API-only targets, IPC channel — all still deferred; rationale in §§8, 10.

### 11.3 Plan deltas and rationale

**(a) EmulatorManager scope grew past the original five methods.**  
Plan named `probe / ensure_booted / shutdown / restore_snapshot / set_proxy`. Delivered adds `install_apk`, `install_apkm`, `save_snapshot`, `grant_permissions`, `push_file`, `get_prop`, `get_density`. Driver: APKMirror ships `.apkm` bundles (split-APK ZIPs), not plain `.apk`; `adb install` cannot consume them directly, so `install_apkm` has to extract, filter splits by ABI/DPI/locale, and call `install-multiple`. That requires `get_prop` (`ro.product.cpu.abi`) and `get_density`. Once the class was touching runtime introspection it was natural to fold in `grant_permissions` (used by `BlackBoxAdapter.run_pipeline` before driving) and `push_file` (used for dataset transport). `set_proxy` became redundant once the proxy port was passed at boot time.

**(b) `primary_backend_host` fallback was built in Phase 1, not Phase 3.**  
Plan put it in "hardening." But every black-box adapter needs a phase boundary; if no LLM host matches (which happens whenever an app proxies inference through its own backend — Replika's architecture, for instance) the classifier degenerates to "everything is POST." The fallback costs ~10 lines and makes the classifier usable for adapters whose backend is non-generic, so it shipped with Phase 1.

**(c) Chrome-on-emulator smoke test was replaced with synthetic-data pytest.**  
Plan wanted a dummy adapter (Chrome → public LLM URL → capture) to validate the observer contract without target-app complexity. In practice the dev machine had no SDK when scaffolding landed, and the observer/driver contract is better protected by unit tests against synthetic traces (easier to run in CI, no emulator spin-up cost, covers branches the Chrome flow would never exercise — e.g. severity-priority log truncation, `primary_backend_host` fallback). `verify/backend/tests/test_blackbox_observers.py` landed with 9 cases, all green.

**(d) `drivers/provision.py` was not in the original plan.**  
Plan assumed one-time manual provisioning per §5 / §12. In practice the cascade of steps (SDK install → `avdmanager create` → headed boot → CA push → APKM install → onboarding click-through → `snapshot save`) is long enough that users drop steps. `provision.py` wraps every non-interactive step around an explicit onboarding pause, keeping the human in the loop only for the clicks. The manual §12 path still works; `provision.py` is the ergonomic version.

**(e) Phase 4 expansion pulled partway into Phase 2.**  
Plan said "port 2–3 targets after end-to-end works." Delivered stubs for Replika and Expensify before the first end-to-end run. Rationale: locator work is per-app and blocks on the AVD, but config (package names, APKM filenames, `primary_backend_host`, `runtime_permissions`, `llm_hosts` augmentation) is pure reading from the target-app READMEs and does not. Landing the config now means the registry, frontend, and `KNOWN_APPS` lists are final; the user's §15 provisioning work can set up all three apps in the same pass rather than returning to scaffolding between each target.

**(f) Frontend badge was pulled from Phase 3 into Phase 2.**  
Plan listed it under "hardening." In practice it cost three lines of conditional caption rendering and made the Initialization page immediately legible once black-box adapters appeared in the registry. No value in deferring it.

**(g) Photomath as Phase 2 target over alternatives.**  
Plan left "easy target" abstract ("receipt scanner, photo caption app"). Expensify would be the natural receipt scanner but its React-Native TLS stack is expected to need `apk-mitm` and possibly Frida (§10 table) — that pushes it into Phase 3 territory. Photomath (Google-owned, OkHttp, moderate pinning) is the lowest-risk target, even though its privacy signal is the weakest of the five APKs. Use it as a tractability benchmark; privacy-meaningful runs live with Replika and Expensify once pinning is handled.

---

## 12. Setup Prerequisites

One-time developer-machine setup before Phase 1:

```bash
# Android SDK + emulator + platform tools
brew install --cask android-commandlinetools
sdkmanager "platform-tools" "emulator" "system-images;android-34;google_apis;arm64-v8a"
avdmanager create avd -n verify_pixel7 -k "system-images;android-34;google_apis;arm64-v8a" --device "pixel_7"

# Python deps (add to verify/requirements.txt)
pip install uiautomator2 mitmproxy

# mitmproxy CA — generated on first run
mitmdump    # Ctrl-C after a second; creates ~/.mitmproxy/*.pem

# APK patching tool (pinning bypass)
npm install -g apk-mitm

# Rooted AVD tooling (optional, for stubborn apps)
# See https://github.com/newbit1/rootAVD
```

Verification:

```bash
emulator -avd verify_pixel7 -no-window -no-audio &
adb wait-for-device
adb shell getprop sys.boot_completed    # → "1" when boot finished
adb emu kill
```

Once this completes cleanly, Phase 1 scaffolding can begin.

---

## 13. Open Questions

These are explicit design decisions to resolve before or during Phase 2:

1. **Serial execution vs. parallel AVDs.** One emulator processes items sequentially. Multiple headless AVDs on different ports could parallelize across items, but AVD memory (~4 GB each) limits fan-out. Tracked alongside the existing orchestrator-parallelization TODO in `verify_report.md` §12.
2. **Snapshot invalidation.** When a target app auto-updates via Play Services, the snapshot drifts. Options: disable Play Services auto-update, pin the installed APK version, re-snapshot on cadence.
3. **Test accounts.** Each app that requires login needs a dedicated test Google / Apple / email account. Build a shared `vendor-accounts/` secrets file (gitignored) with credentials per adapter.
4. **Dataset item transport.** Image items need to land on the emulator filesystem. Use `adb push` to `/sdcard/Download/` before each run; clean up after. Text items are typed via `set_text`. Video items may need file-chooser interception (per-app specifics).
5. **Response extraction reliability.** `d(...).get_text(timeout=60)` works when the response lands in a known view. Streaming UIs (text appearing token-by-token) need polling; chat UIs with message bubbles need the last-message locator. Handled per-adapter in `_drive_app`.

---

## 14. Implementation Progress Log

Running log of what has been built in the repo. Update this section as phases land.

### 14.1 Closed-source target apps added (`target-apps/`)

Five APKs staged under `target-apps/<name>/` (binaries gitignored, metadata committed):

| Directory | Package | Version | Build | Format | ABI splits | Difficulty |
|---|---|---|---|---|---|---|
| `photomath/` | `com.microblink.photomath` | 8.47.1 | 70001015 | `.apkm` | 4arch × 7dpi | Low-Medium |
| `replika/` | `ai.replika.app` | 12.7.2 | 6340 | `.apkm` | 2arch × 7dpi + 1 feature | Medium |
| `expensify/` | `org.me.mobiexpensifyg` | 9.3.58-9 | 509035809 | `.apkm` | 4arch × 7dpi | Medium-High |
| `noom/` | `com.wsl.noom` | 14.12.0 | 302330 | `.apkm` | 4arch × 7dpi + 1 feature | **Dropped — not in registry** (paywall gate) |
| `google-lens/` | `com.google.ar.lens` | 1.18.250731009 | 250731009 | `.apk` | arm64-v8a only | **Dropped — not in registry** (Play Integrity / attestation risk) |

Each directory has a `README.md` with package info, install commands, login/pinning expectations, privacy relevance, and recommended dataset.

**Gitignore additions** (`/.gitignore`):
```
target-apps/*/*.apk
target-apps/*/*.apkm
target-apps/*/_split/
```

**`.apkm` handling.** APKMirror bundles are ZIP archives of split APKs. `adb install foo.apkm` fails; the correct path is `unzip` → filter splits matching AVD ABI + DPI + locales → `adb install-multiple`. Automated in `EmulatorManager.install_apkm()`.

### 14.2 Phase 1 scaffolding — code landed

All new modules under `verify/backend/`. Syntax- and import-verified; synthetic-flow unit test for `phase_classifier` passes.

#### `drivers/` (new package)

**`drivers/emulator_manager.py`** — AVD lifecycle manager.

| Method | Purpose |
|---|---|
| `probe(avd_name)` | Non-blocking: checks `emulator` + `adb` on PATH (with `ANDROID_HOME` fallback), verifies AVD exists |
| `ensure_booted(headless=True)` | Reuses running emulator by AVD name, else spawns `emulator -avd … -no-audio -no-window`; waits up to 180 s for `sys.boot_completed=1`. Intentionally does **not** pass `-http-proxy`, because a boot-time proxy black-holes traffic whenever mitmdump isn't running (provisioning, onboarding, snapshot save). |
| `set_runtime_proxy(port)` / `clear_runtime_proxy()` | Toggle `adb shell settings put global http_proxy 10.0.2.2:<port>`. Called inside `run_pipeline` only while `NetworkObserver` is active. |
| `restore_snapshot(name)` | `adb emu avd snapshot load <name>`; non-fatal if missing |
| `save_snapshot(name)` | `adb emu avd snapshot save <name>` |
| `install_apk(path)` | `adb install -r -t <apk>` |
| `install_apkm(path, locales=("en",))` | Extracts ZIP, filters splits to match `ro.product.cpu.abi` + density bucket + locales + feature modules, calls `install-multiple` |
| `grant_permissions(package, perms)` | `pm grant` per permission |
| `push_file(local, remote)` | `adb push` |
| `get_prop(key)` / `get_density()` | Runtime introspection |
| `shutdown()` | `adb emu kill` + terminate the `emulator` subprocess |

Proxy port is auto-picked from a free port in `8082–8181` at manager construction time.

#### `observers/` (new package)

**`observers/_mitm_addon.py`** — runs inside `mitmdump`, stdlib-only + `mitmproxy`. On every completed flow, appends a JSON record to the file named by `MITM_FLOW_LOG`:

```
{ts, method, scheme, host, port, path, url,
 status, req_content_type, req_bytes, res_content_type, res_bytes,
 req_body, res_body}
```

Response bodies retained only for `application/json` content-types; truncated at 4 KB.

**`observers/network_observer.py`** — `NetworkObserver(proxy_port)` context manager.
- `start()` spawns `mitmdump -p <port> -s _mitm_addon.py -q` with `MITM_FLOW_LOG` pointing at a temp file. Waits up to 10 s for the port to bind.
- `stop()` sends `SIGINT` → `SIGTERM` → `SIGKILL` with escalating timeouts, then parses the JSON-lines log into `self.events` (sorted by timestamp).
- `.flows` property is an alias for `.events`.

**`observers/phase_classifier.py`** — replaces `_runtime_capture.set_phase("POST")` without source access.

- `DEFAULT_LLM_HOSTS` — 14 fnmatch patterns: `*.openai.com`, `*.anthropic.com`, `generativelanguage.googleapis.com`, `openrouter.ai`, `api.together.xyz`, `api.mistral.ai`, `api.groq.com`, `*.cohere.ai`, `api.replicate.com`, `api.x.ai`, `api.perplexity.ai`, `api.deepseek.com`, `*.azure.com`, and a broad `*.googleapis.com` minus an explicit deny-list (Firebase, Play Services, Android, crash reporting, people-pa).
- `classify_phases(events, llm_hosts, primary_backend_host)` → `(during, post)`. Boundary `t*` is the timestamp of the **last LLM-host response**; fallback is first response from `primary_backend_host`; if neither, everything is treated as POST.
- Events with `ts < t*` → DURING (discarded); `ts ≥ t*` → POST (returned as externalizations).

Unit-tested with a synthetic 4-flow trace (login → OpenAI → Firebase → analytics); correctly partitions 1 → DURING and 3 → POST.

#### `adapters/blackbox_base.py` (new base class)

**`BlackBoxConfig`** dataclass fields: `package_name`, `main_activity`, `apk_filename` / `apkm_filename`, `avd_name`, `snapshot_name`, `llm_hosts`, `primary_backend_host`, `runtime_permissions`, `timeout_s`.

**`BlackBoxAdapter(BaseAdapter)`** — concrete subclasses override `_drive_app(driver, input_item) → str` and set `config`. Base class implements:

- `check_availability()` — returns `(False, ...)` if `USE_APP_SERVERS=false` (no serverless fallback for closed-source in v1); otherwise delegates to `EmulatorManager.probe`.
- `run_pipeline(input_item)` — full orchestration:
    1. Modality guard.
    2. `EmulatorManager.ensure_booted()` → attach or spawn.
    3. `restore_snapshot` (soft-fail with note in metadata).
    4. Runtime-permission grants.
    5. Enter `NetworkObserver` context, lazy-import `AndroidDriver`, call subclass `_drive_app`.
    6. Exit observer → `classify_phases(events, llm_hosts, primary_backend_host)`.
    7. Return `AdapterResult` with POST-phase externalizations flattened via `_flatten_post()` (first 15 network events + response-body previews).
- `_flatten_post()` reserves slots for `STORAGE` / `LOGGING` / `UI`; empty until Phase 2 observers land.

### 14.3 Verification run

```
$ python -c "from verify.backend.observers import NetworkObserver, …"
imports OK
  DEFAULT_LLM_HOSTS has 14 patterns
  EmulatorManager.probe(nonexistent): ok=False
  message: [BLACKBOX] 'emulator' not on PATH — install Android SDK + set ANDROID_HOME...
```

Good failure mode when Android SDK is absent (expected on this dev machine).

### 14.4 Phase 2 code landed

All new modules under `verify/backend/`. Parse- and import-verified; synthetic unit tests for `UiObserver.stop()`, `LogObserver._parse_lines()`, and `classify_phases()` (all four channels) pass.

#### `drivers/android_driver.py`

Thin `uiautomator2` wrapper matching the API referenced by `BlackBoxAdapter.run_pipeline`.

| Method | Purpose |
|---|---|
| `launch()` / `stop()` | Cold-start + force-stop the target package; uses `activity=""` to fall back to the default launch activity |
| `tap(locator, timeout)` | Click the first match (`wait` before `click`) |
| `type_into(locator, text, timeout)` | Set element text |
| `read(locator, timeout)` | Wait then return stripped `.text`; raises `TimeoutError` if not found |
| `wait_for(locator, timeout)` / `exists(locator)` | Blocking / non-blocking presence checks |
| `push_image(local_path, remote_path)` | `adb push` then `MEDIA_SCANNER_SCAN_FILE` broadcast so galleries pick up the new file immediately |
| `screenshot(path)` / `dump_hierarchy()` | Diagnostics; the latter is what `UiObserver` consumes |
| `press_back()` / `press_home()` | Hardware-key shortcuts for dialog dismissal |

`uiautomator2` is imported lazily via a `.d` property so the module stays importable on dev machines without the Android SDK.

#### `observers/fs_observer.py`

Before/after stat listing of `/data/data/<pkg>` and `/sdcard`, diffed into STORAGE events `{ts, kind, path, size, mtime, ...}`.

- `_list(root)` runs `find … -printf "%T@|%s|%p"` first (works on `/sdcard` and on rooted AVDs for `/data/data`); falls back to `run-as <pkg> find` for debuggable apps on user-build AVDs.
- Noise filter: `/cache/`, `/code_cache/`, `/tmp/`, `.lock`, `.tmp`, `.log`, `.pid`.
- Ranking: files matching interesting suffixes (`.json/.db/.sqlite*/.jpg/.png/.pdf/.txt/.csv/.xml/.proto/.bin`) float to the top; total capped at 200.

#### `observers/log_observer.py`

Background `adb logcat -v threadtime --uid <app_uid>` subprocess; on stop, parses lines with a regex matching `MM-DD HH:MM:SS.mmm PID TID L TAG : MSG` and emits LOGGING events.

- UID lookup via `cmd package list packages -U <pkg>` so the filter is tight even if the app writes under system tags.
- Buffer cleared (`logcat -c`) on `start()` so we only see this run.
- Severity-priority truncation: WARN+/ERROR/FATAL kept first, lower levels fill remaining budget up to 200 records (matches the spirit of `_runtime_capture.py`'s logging filter).

#### `observers/ui_observer.py`

Two `d.dump_hierarchy()` snapshots per run: baseline at `start()`, post-response on `capture_post()` (or `stop()` as fallback). Hierarchy XML is parsed with `ElementTree`; each `<node>` is reduced to `(text, content-desc, resource-id, class)` and keyed as `"{rid}|{text}|{desc}"`.

Any post-run key not in the baseline set becomes a UI event:
- non-empty `text` (≥2 chars) → `{kind: "text"}`
- else non-empty `content-desc` → `{kind: "desc"}`
- else `ImageView` node with a resource-id → `{kind: "image"}`

Soft-fails (empty events, no exception) if `uiautomator2` is unavailable so the other observers still capture.

#### `adapters/blackbox_base.py` — observers wired in

`run_pipeline` now enters all four observers in one `with` stack:

```python
net = NetworkObserver(proxy_port=em.proxy_port)
fs  = FsObserver(serial=em.serial, package=…)
log = LogObserver(serial=em.serial, package=…)
ui  = UiObserver(serial=em.serial)

with net, fs, log, ui:
    driver.launch()
    response_text = self._drive_app(driver, input_item)
    ui.capture_post()
    driver.stop()
```

`_flatten_post()` now formats all four POST channels: NETWORK (method/url/status + 200-char body), STORAGE (kind/path/size, up to 20), LOGGING (level/tag/msg, up to 30), UI (kind/value, up to 30). Metadata returns `n_*_events` counts for each channel.

#### `adapters/photomath_blackbox.py` — first concrete adapter

Registered as `"photomath"` in `ADAPTER_REGISTRY`.

- `BlackBoxConfig`: `com.microblink.photomath`, default launcher activity, `apkm_filename` pointing at the staged APKM, `snapshot_name="clean"`, `llm_hosts = DEFAULT_LLM_HOSTS + (*.photomath.net, api.photomath.com)`, `primary_backend_host="*.photomath.net"`, runtime permissions for media read.
- `_drive_app(driver, item)`: `push_image` → `/sdcard/DCIM/verify_input.jpg`, best-effort dismissal of any surviving onboarding/permission dialogs, gallery-import button via `descriptionContains`/`resourceIdMatches`, pick the pushed image, read `id/problem*` + `id/solution*` text views. Returns `"Problem: … / Solution: …"`.

Locator dicts are best-effort for Photomath 8.47.1 and will need verification against a `dump_hierarchy` at snapshot-provisioning time (§5).

### 14.5 Verification runs

- Observer imports green; 9/9 pytest cases in `test_blackbox_observers.py`.
- First real on-device run (2026-04-19, Photomath 8.47.1, `verify_pixel7`, snapshot `clean`): `AdapterResult.success=True`, 12 UI events, 30 LOGGING events captured. NETWORK/STORAGE 0 because the flow stopped at the Android photo-picker (see §15.9).

### 14.6 Still not implemented (Phase 3+)

| Component | Phase | Status |
|---|---|---|
| Photomath photo-picker + solver locators (second-half of `_drive_app`) | 2 | Gallery button done; picker-thumbnail + `problem/solution` locators pending (§15.9–15.10) |
| Pinning bypass helpers (`apk-mitm`, Frida) | 3 | Partial — `drivers/pinning.py` landed with `apk-mitm` patching for single APKs and APKM base.apk + split re-signing support; `provision.py --apk-mitm` now installs consistently re-signed split bundles. Replika is marked `pinning_bypass="apk_mitm"`. Runtime stability is still unresolved (§15.3). |
| Frontend wiring (`KNOWN_APPS` entries, device-required badge, emulator status indicator) | 3 | Not started |
| Replika / Expensify `_drive_app` implementations | 4 | Stubs raise `NotImplementedError` |

### 14.7 Target-scope decisions

Originally five APKs were staged; three are now in scope and two are dropped.

**In scope (registered in `ADAPTER_REGISTRY`):**

| App | Adapter | Status |
|---|---|---|
| Photomath | `PhotomathAdapter` | First run succeeded 2026-04-19 with UI+LOGGING events; gallery button locator patched. Photo-picker + solver locators pending (§15.9–15.10). |
| Replika | `ReplikaAdapter` | Config populated and now marked `pinning_bypass="apk_mitm"`. Patched split install succeeds, but the patched app still hangs/ANRs on startup before reaching a usable chat UI (§15.3). `_drive_app` still raises `NotImplementedError`. |
| Expensify | `ExpensifyAdapter` | Config populated; `_drive_app` raises `NotImplementedError` until locators verified. Will likely need `apk-mitm` for TLS pinning. |

**Dropped (not registered, binaries remain staged under `target-apps/` in case we revisit):**

| App | Reason |
|---|---|
| Noom | Meal-logging flow requires an active paid subscription; no realistic path to a reproducible test account. Replace with a free tracker if this use case becomes important later. |
| Google Lens | Play Integrity / hardware attestation likely refuses any AVD. Not worth the cat-and-mouse work before the rest of the stack is mature (§9 table). Revisit in Phase 5 if at all. |

Phase 2 target order, with Noom and Google Lens removed: **Photomath → Replika → Expensify**.

---

## 15. Next Actions (for the user)

### ✅ Completed (15.1–15.7)

Environment setup, AVD provisioning (`verify_pixel7`), APKM install, onboarding click-through, `clean` snapshot, dataset build (`batch_config_photomath.csv`), and first end-to-end run are all done.

**First run summary** (2026-04-19, `00_2x+3=11.png`, 2:07 wall): `success=True`, 12 UI + 30 LOGGING events. Gallery button matched via `gallery_fragment_container → ImageButton` child selector. Flow stalled at Android 14 scoped photo-picker — no NETWORK/STORAGE because the solver screen was never reached.

Run command (for reference):
```bash
adb -s emulator-5554 emu avd snapshot load clean
USE_APP_SERVERS=true python verify/run_batch.py \
    --config verify/batch_config_photomath.csv \
    --mode ioc --workers 1 --item-workers 1 --no-cache
```

---

### ✅ 15.1 Photo-picker + solver locators confirmed (2026-04-20)

Hierarchy dump with image present revealed the full flow:

- **MediaStore insert:** `adb shell content insert --uri content://media/external/images/media` is synchronous and reliable (replaces deprecated `MEDIA_SCANNER_SCAN_FILE` broadcast).
- **Thumbnail locator:** `descriptionContains="Photo taken on"` (Android 14 labels thumbnails by timestamp). No confirmation button — picker returns immediately on single tap.
- **Crop screen:** after thumbnail tap, Photomath shows a crop-adjustment screen with `button_solve`.
- **Solver screen:** `solution_card_container`; `card_title` = problem type, `card_header` = solution type.

### ⏸ 15.2 STALLED: picker auto-dismisses when proxy is active (Photomath)

**Root cause (confirmed via `adb screencap`):** the Android 14 scoped photo-picker closes itself within ~1 second of opening when mitmproxy is active. The picker makes a network request (Google Photos load) that goes through the proxy; the TLS intercept causes an error and the picker auto-dismisses. The thumbnail IS visible at t=1s but gone by t=2s.

**What was validated on 2026-04-24:**

- The home-screen gallery button locator is correct: `gallery_fragment_container → ImageButton`.
- With proxy **off**, the picker stays open and exposes both:
  - `descriptionContains="Photo taken on"` on the first thumbnail tile
  - `resourceId="com.google.android.providers.media.module:id/icon_thumbnail"` on the thumbnail image
- With proxy **off**, a manual thumbnail tap returns to the crop screen reliably; the crop screen shows `button_solve`.
- With proxy re-enabled **after** the crop screen appears, tapping **Solve** reaches Photomath's own network-dependent path and produces the in-app "can't connect" / "try again" error rather than dismissing the picker. So the picker and solver failures are distinct.

**Adapter changes landed on 2026-04-24 (`adapters/photomath_blackbox.py`):**

- Removed the old hard-coded thumbnail coordinate tap (`input tap 177 825`).
- Added proxy helpers (`_get_proxy`, `_clear_proxy`, `_set_proxy`) so the adapter can temporarily drop the global MITM proxy only for the picker.
- Switched picker selection to real thumbnail locators (`descriptionContains="Photo taken on"` with a fallback to `com.google.android.providers.media.module:id/icon_thumbnail`) plus bounds-center tapping.

**Current status after those changes:**

- The picker branch is narrower than before but still flaky under the full `run_pipeline()` path after `restore_snapshot("clean")`.
- Successful direct-device repros do **not** yet translate into a stable end-to-end adapter run. Some runs still return from the picker without selecting the image; one run reached the picker and then falsely read picker-sheet text as `output_text`.
- The current stable claim is therefore: *proxy-clearing around the picker is necessary, but not yet sufficient for a reproducible Photomath artifact run.*

**Intent-bypass experiments (2026-04-24):**

- `am start -a android.intent.action.VIEW -d content://media/external/images/media/<id> -t image/jpeg com.microblink.photomath`
  does **not** resolve on this build. `cmd package resolve-activity` reports `No activity found`.
- `dumpsys package com.microblink.photomath` shows that the APK registers `android.intent.action.SEND` for `image/*` on `.main.activity.LauncherActivity`, not `VIEW` for `content://...` media URIs.
- Forcing `SEND image/jpeg` with `android.intent.extra.STREAM=content://media/.../<id>` launches Photomath, but the app immediately shows:
  `Oops! Something went wrong! Please try again.`
- Conclusion: the previously suggested `VIEW`-intent bypass is not a valid fallback for this APK version, and the `SEND`-intent path is presently not usable either.

**Pending fix options (pick one when returning to Photomath):**

- **A. Clear proxy around picker** — call `em.clear_runtime_proxy()` before `gallery.click()`, then re-set after thumbnail is selected. Cleaner; loses a small window of NETWORK capture.
- **B. Post-picker snapshot** — manually navigate to the crop screen once, save a `post_picker` snapshot. Batch restores it instead of `clean`. Loses gallery-flow NETWORK events.
- **C. Rooted physical-device variant** — revisit on a rooted real phone if the emulator's system picker remains unstable under instrumentation. This is a separate execution mode, not a drop-in replacement for the AVD path.

**Moving on to Replika first.**

### 15.3 TODO: Phase 3 hardening (unblocks Replika + Expensify)

Deferred until §15.1–15.2 produce non-trivial NETWORK + solver output.

- [x] `apk-mitm` wrapper — landed on 2026-04-24. `drivers/pinning.py` can patch a single APK or patch `base.apk` from an APKM bundle, then re-sign all selected splits with the local Android debug keystore so `adb install-multiple` accepts the bundle.
- [ ] Frida fallback for hard-pinned apps.
- [ ] Implement `_drive_app` for `replika_blackbox.py` (text chat flow) and `expensify_blackbox.py` (receipt image flow) — currently both raise `NotImplementedError`.
- [ ] `ObserverPipeline` wrapper to replace the inline `with net, fs, log, ui:` in `blackbox_base.py`.

**Replika hardening status (2026-04-24):**

- Vanilla Replika was not installed in the current `clean` snapshot; it had to be installed explicitly onto the running AVD.
- The current launcher/onboarding surface is reachable, but the snapshot is not yet in the saved-session chat state assumed by `replika_blackbox.py`.
- Applying `apk-mitm` to Replika's `base.apk` succeeds and reports a concrete pinning-related patch:
  `ai/replika/app/ev9: Applied HostnameVerifier#verify (javax) patch`
- Replacing only `base.apk` is insufficient for split bundles: `adb install-multiple` rejects mixed-signature installs with
  `INSTALL_FAILED_INVALID_APK ... signatures are inconsistent`.
- Re-signing the entire selected split set with the Android debug key fixes the install problem; the patched split bundle installs successfully on the emulator.
- The patched Replika build then launches into `ai.replika.app/.home.ui.MainActivity`, but runtime stability is still poor:
  - the app can stall on the black splash / onboarding surface
  - Android may surface a "Replika isn't responding" dialog
  - `logcat` shows repeated warnings from `ai.replika.app` of the form
    `No such thread for suspend: ... :main`
- Current conclusion: **install-layer hardening for Replika is working; runtime hardening is not yet sufficient.** The next escalation, if Replika remains a priority, is a Frida-based bypass rather than more APKM install work.

### 15.4 Unblocking failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `check_availability()` returns "emulator not on PATH" | `ANDROID_HOME` not exported | Re-export in the shell that runs Python |
| AVD has no internet during onboarding | Stale `settings global http_proxy` pointing at dead mitmdump port | `adb shell settings delete global http_proxy` + `adb emu kill` + reboot via `provision.py` |
| mitmdump never binds port | Port conflict | `lsof -i :8082`; kill stragglers |
| App sees no network | System CA not trusted | Re-verify `/system/etc/security/cacerts/` push survived reboot |
| `UI` events empty | Locators drifted / solver screen never reached | Re-dump hierarchy, update `_drive_app` |
| `STORAGE` events empty | `/data/data/com.microblink.photomath` unreadable | Confirm `adb root` took; fall back to `/sdcard` diff |
| `LogObserver` has 0 records | App UID mismatch or buffer cleared too late | Drop `--uid` filter in `log_observer.py` temporarily |
