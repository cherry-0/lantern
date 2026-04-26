# Useful Commands

Practical command reference for working with Verify in this repository.

This file complements:
- `README.md` for setup and high-level usage
- `analysis/verify_report.md` for architecture and implementation details

---

## 1. Installation

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url> lantern
cd lantern
```

If submodules were not cloned initially:

```bash
git submodule update --init --recursive
```

Create the main Verify host environment:

```bash
conda create -n lantern python=3.12
conda activate lantern
pip install -r verify/requirements.txt
```

Optional bulk install for native target-app dependencies:

```bash
./install_all_reqs.sh
```

---

## 2. Environment Setup

Create `.env` at repo root with at least:

```env
OPENROUTER_API_KEY=sk-or-...
USE_APP_SERVERS=false
DEBUG=false
VERBOSE=false
```

Typical modes:

```bash
# Serverless mode: fastest path, uses OpenRouter only
export USE_APP_SERVERS=false

# Native mode: runs real app pipelines in isolated conda envs
export USE_APP_SERVERS=true
```

---

## 3. Launch the Streamlit App

```bash
conda activate lantern
streamlit run verify/frontend/app.py
```

---

## 4. Initialization

Use the Streamlit `Initialization` page for per-app env setup, or initialize environments lazily by running the pipeline in native mode.

If you only want serverless mode:

```bash
export USE_APP_SERVERS=false
```

If you want native mode:

```bash
export USE_APP_SERVERS=true
```

---

## 5. Batch Pipeline

Run all enabled rows from `verify/batch_config.csv`:

```bash
python verify/run_batch.py
```

IOC only:

```bash
python verify/run_batch.py --mode ioc
```

Perturbation only:

```bash
python verify/run_batch.py --mode perturb
```

Custom config file:

```bash
python verify/run_batch.py --config verify/batch_config.csv --mode both
```

Parallelize across config rows:

```bash
python verify/run_batch.py --workers 4
```

Limit dataset items per run:

```bash
python verify/run_batch.py --max-items 20
```

Preview without executing:

```bash
python verify/run_batch.py --dry-run
```

Disable cache reuse:

```bash
python verify/run_batch.py --no-cache
```

---

## 6. Re-evaluation

Stamp provenance on existing cached results:

```bash
python verify/reeval.py --init
```

Re-evaluate all output directories with a model:

```bash
python verify/reeval.py --model google/gemini-2.5-pro
```

Re-evaluate only selected apps or datasets:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --app deeptutor xend
python verify/reeval.py --model google/gemini-2.5-pro --dataset PrivacyLens
```

Re-evaluate specific directories:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --dir verify/outputs/cache_d2fbdc5307a7c57b
```

Bare output-directory names are also accepted:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --dir cache_d2fbdc5307a7c57b
```

Dry-run preview:

```bash
python verify/reeval.py --dry-run --model google/gemini-2.5-pro
```

### Prompt variants

Default binary prompt:

```bash
python verify/reeval.py --model google/gemini-2.5-pro
```

MCQ/value-prediction prompt:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt2
```

Channel-wise + aggregate threat prompt:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt3
```

Prompt3 on one cache:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt3 --dir cache_d2fbdc5307a7c57b
```

### Parallel re-eval

Parallelize within a directory:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --workers 4
```

Parallelize across directories too:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --workers 4 --dir-workers 2
```

Two-level parallelism on selected caches:

```bash
python verify/reeval.py \
  --model google/gemini-2.5-pro \
  --prompt3 \
  --workers 4 \
  --dir-workers 2 \
  --dir cache_d2fbdc5307a7c57b cache_44a4eac084755540
```

Note: total OpenRouter concurrency is approximately `workers * dir-workers`.

---

## 7. Evaluation Validation

Populate MCQ predictions for SynthPAI:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt2 --dataset SynthPAI
```

Then open the Streamlit `Eval Validation` page to compare:
- `ext_eval[attr].inferable` vs. ground truth
- `ext_eval[attr].prediction` vs. SynthPAI profile values

---

## 8. IOC / Stage-wise Analysis

Open the Streamlit app and use:
- `Input-Output Comparison` to run IOC live
- `View Input-Output Comparison Results` to inspect cached IOC results

Useful command before opening those pages:

```bash
streamlit run verify/frontend/app.py
```

If you want prompt3-style channel-wise externalization scoring on an IOC cache:

```bash
python verify/reeval.py --model google/gemini-2.5-pro --prompt3 --dir cache_d2fbdc5307a7c57b
```

---

## 9. Cache and Output Inspection

List output directories:

```bash
ls verify/outputs
```

Inspect a run config:

```bash
cat verify/outputs/<run_or_cache_dir>/run_config.json
```

Inspect output-directory summary metadata:

```bash
cat verify/outputs/<run_or_cache_dir>/dir_summary.json
```

Inspect one cached item:

```bash
cat verify/outputs/<run_or_cache_dir>/row_00000.json
```

---

## 10. Sanity Checks

Syntax-check key scripts:

```bash
python -m py_compile verify/reeval.py
python -m py_compile verify/run_batch.py
python -m py_compile verify/frontend/pages/6_Reeval.py
```

Check that `conda` is available:

```bash
conda --version
```

Check that Streamlit can import:

```bash
python -c "import streamlit; print(streamlit.__version__)"
```

---

## 11. Recommended Workflows

Fast serverless workflow:

```bash
conda activate lantern
export USE_APP_SERVERS=false
streamlit run verify/frontend/app.py
```

Batch + re-eval workflow:

```bash
conda activate lantern
python verify/run_batch.py --workers 4
python verify/reeval.py --init
python verify/reeval.py --model google/gemini-2.5-pro --prompt3 --workers 4 --dir-workers 2
```

SynthPAI validation workflow:

```bash
conda activate lantern
python verify/run_batch.py --mode ioc --workers 4
python verify/reeval.py --model google/gemini-2.5-pro --prompt2 --dataset SynthPAI
streamlit run verify/frontend/app.py
```

---

## 12. Android Device and Emulator Commands

These commands are useful when testing closed-source Android apps such as
Photomath and Expensify on a real device or emulator.

Use a shell variable for the device serial:

```bash
export SERIAL=9A311FFAZ002PU
```

For an emulator, the serial usually looks like:

```bash
export SERIAL=emulator-5554
```

### Device discovery

List connected devices:

```bash
adb devices
```

List devices with model/product metadata:

```bash
adb devices -l
```

Check CPU ABI and screen density before installing split APKs:

```bash
adb -s "$SERIAL" shell getprop ro.product.cpu.abi
adb -s "$SERIAL" shell wm density
```

Common mapping for APKMirror split bundles:

```text
arm64-v8a  -> split_config.arm64_v8a.apk
440 dpi    -> split_config.xxhdpi.apk
```

### Installing apps

Install a single APK:

```bash
adb -s "$SERIAL" install -r -t path/to/app.apk
```

Install a split APK set:

```bash
adb -s "$SERIAL" install-multiple -r -t \
  target-apps/photomath/_split/base.apk \
  target-apps/photomath/_split/split_config.arm64_v8a.apk \
  target-apps/photomath/_split/split_config.xxhdpi.apk
```

Expensify example:

```bash
adb -s "$SERIAL" install-multiple -r -t \
  target-apps/expensify/_split/base.apk \
  target-apps/expensify/_split/split_config.arm64_v8a.apk \
  target-apps/expensify/_split/split_config.xxhdpi.apk
```

Check whether a package is installed:

```bash
adb -s "$SERIAL" shell pm path com.microblink.photomath
adb -s "$SERIAL" shell pm path org.me.mobiexpensifyg
```

Uninstall a package:

```bash
adb -s "$SERIAL" uninstall com.microblink.photomath
```

Grant runtime permissions when supported by the device OS:

```bash
adb -s "$SERIAL" shell pm grant com.microblink.photomath android.permission.CAMERA
adb -s "$SERIAL" shell pm grant com.microblink.photomath android.permission.READ_EXTERNAL_STORAGE
```

Note: newer permissions such as `android.permission.READ_MEDIA_IMAGES` only
exist on newer Android versions. If `pm grant` says `Unknown permission`, use
the older storage permission or grant through the UI.

### Launching and stopping apps

Launch an app through its launcher activity:

```bash
adb -s "$SERIAL" shell monkey -p com.microblink.photomath -c android.intent.category.LAUNCHER 1
```

Force-stop an app:

```bash
adb -s "$SERIAL" shell am force-stop com.microblink.photomath
```

Return to home:

```bash
adb -s "$SERIAL" shell input keyevent KEYCODE_HOME
```

Press back:

```bash
adb -s "$SERIAL" shell input keyevent KEYCODE_BACK
```

### Screen capture

Preferred screenshot method when broad `adb` is allowed:

```bash
adb -s "$SERIAL" shell screencap -p /sdcard/screen.png
adb -s "$SERIAL" pull /sdcard/screen.png /tmp/screen.png
```

This avoids host-side redirection and is usually easier to run in sandboxes
than `exec-out ... > /tmp/file.png`.

Direct host screenshot method:

```bash
adb -s "$SERIAL" exec-out screencap -p > /tmp/screen.png
```

Use direct `exec-out` when you specifically need a one-command capture and the
environment allows shell redirection.

### UI hierarchy inspection

Dump current UI XML to the device:

```bash
adb -s "$SERIAL" shell uiautomator dump /sdcard/window.xml
adb -s "$SERIAL" pull /sdcard/window.xml /tmp/window.xml
```

Search useful locator fields:

```bash
rg -o 'text="[^"]*"|resource-id="[^"]*"|content-desc="[^"]*"' /tmp/window.xml
```

Look for specific controls:

```bash
rg -n 'gallery|Solve|Scan|receipt|Expenses|Sign in|Continue|Skip' /tmp/window.xml
```

Useful locator fields for adapter code:

```text
text="Solve"
resource-id="com.microblink.photomath:id/gallery_fragment_container"
content-desc="Preview the file verify_input.png"
class="android.widget.ImageButton"
bounds="[338,1973][482,2117]"
```

### Tap, swipe, and text input

Tap a screen coordinate:

```bash
adb -s "$SERIAL" shell input tap 728 1085
```

Swipe from one coordinate to another:

```bash
adb -s "$SERIAL" shell input swipe 540 2050 540 1250 500
```

Use cases:

```bash
# Dismiss a dialog X button.
adb -s "$SERIAL" shell input tap 923 967

# Tap Photomath's Solve button on the crop screen.
adb -s "$SERIAL" shell input tap 728 1085

# Open Photomath's gallery button near the lower-left camera controls.
adb -s "$SERIAL" shell input tap 410 2045

# Scroll Photomath result cards upward to reveal solving steps.
adb -s "$SERIAL" shell input swipe 540 2050 540 1250 500

# Pan an image right under Photomath's crop box when the left side is cut off.
adb -s "$SERIAL" shell input swipe 220 710 450 710 700
```

Type text into the focused field:

```bash
adb -s "$SERIAL" shell input text 'hello%sworld'
```

`%s` inserts a space.

### Media files and gallery pickers

Push an image to the device:

```bash
adb -s "$SERIAL" push verify/backend/datasets/photomath_math/04_12*7.png /sdcard/DCIM/verify_input_12x7.png
```

Make it visible to Android file and gallery pickers through MediaStore:

```bash
adb -s "$SERIAL" shell content insert \
  --uri content://media/external/images/media \
  --bind _display_name:s:verify_input_12x7.png \
  --bind _data:s:/storage/emulated/0/DCIM/verify_input_12x7.png \
  --bind mime_type:s:image/png \
  --bind is_pending:i:0
```

Some Android builds support media scanning:

```bash
adb -s "$SERIAL" shell cmd media scan /sdcard/DCIM/verify_input_12x7.png
```

If this returns `cmd: Can't find service: media`, use the MediaStore
`content insert` command above.

### Network and proxy state

Check Wi-Fi/network state from the UI when apps block on connectivity:

```bash
adb -s "$SERIAL" shell uiautomator dump /sdcard/window.xml
adb -s "$SERIAL" pull /sdcard/window.xml /tmp/window.xml
rg -n 'Internet|No networks available|Wi-Fi|Done' /tmp/window.xml
```

Clear a stale Android global HTTP proxy:

```bash
adb -s "$SERIAL" shell settings put global http_proxy :0
adb -s "$SERIAL" shell settings delete global http_proxy
```

Set a proxy for emulator MITM traffic:

```bash
adb -s "$SERIAL" shell settings put global http_proxy 10.0.2.2:8082
```

Set a proxy for a real device on the same LAN as the host:

```bash
adb -s "$SERIAL" shell settings put global http_proxy <host-lan-ip>:8082
```

### App-specific notes from real-device testing

Photomath:

```text
Package: com.microblink.photomath
Install type: split APK set from target-apps/photomath/_split
Gallery control: com.microblink.photomath:id/gallery_fragment_container
Crop screen: manually include the whole math expression before tapping Solve
Observed result: 12 x 7 -> 84 after panning the image into crop coverage
```

Real-device privacy-evaluation status:

```text
Available today:
- Manual or semi-automated UI execution on a real device with adb.
- UI output capture through screenshots and uiautomator XML dumps.
- Log capture through adb logcat.
- Shared-storage diffing under /sdcard.

Not yet equivalent to the emulator batch pipeline:
- verify/run_batch.py -> Photomath currently goes through BlackBoxAdapter,
  which assumes EmulatorManager, an AVD snapshot, and emulator proxy semantics.
- /data/data/<package> storage diffing requires root or run-as; production
  real devices usually only expose /sdcard.
- NetworkObserver uses mitmdump; a real device needs a LAN proxy, trusted CA,
  and possibly apk-mitm/frida if the app pins TLS.

Practical conclusion:
- Use the adb recipe below for repeatable real-device Photomath UI/output runs.
- Use the existing Verify black-box batch path for emulator-based automated
  privacy-leak scoring until a RealDeviceManager runner is added.
```

Run the existing emulator-oriented Photomath batch experiment:

```bash
export USE_APP_SERVERS=true
python verify/run_batch.py --config verify/batch_config_photomath.csv --mode both --workers 1 --max-items 1
```

Real-device Photomath manual run:

```bash
export SERIAL=9A311FFAZ002PU

# Start clean.
adb -s "$SERIAL" shell am force-stop com.microblink.photomath
adb -s "$SERIAL" shell monkey -p com.microblink.photomath -c android.intent.category.LAUNCHER 1

# Push an input image and register it with MediaStore.
adb -s "$SERIAL" push verify/backend/datasets/photomath_math/04_12*7.png /sdcard/DCIM/verify_input_12x7.png
adb -s "$SERIAL" shell content insert \
  --uri content://media/external/images/media \
  --bind _display_name:s:verify_input_12x7.png \
  --bind _data:s:/storage/emulated/0/DCIM/verify_input_12x7.png \
  --bind mime_type:s:image/png \
  --bind is_pending:i:0

# Open gallery picker from Photomath camera screen.
adb -s "$SERIAL" shell input tap 410 2045

# In Android DocumentsUI, tap the desired recent image thumbnail.
# Coordinates depend on screen layout; inspect first:
adb -s "$SERIAL" shell screencap -p /sdcard/photomath_picker.png
adb -s "$SERIAL" pull /sdcard/photomath_picker.png /tmp/photomath_picker.png

# If the image is partly outside the crop coverage, pan the image under the crop box.
adb -s "$SERIAL" shell input swipe 220 710 450 710 700

# Tap Solve.
adb -s "$SERIAL" shell input tap 728 1085

# Capture output for evaluation / manual inspection.
adb -s "$SERIAL" shell screencap -p /sdcard/photomath_result.png
adb -s "$SERIAL" pull /sdcard/photomath_result.png /tmp/photomath_result.png
adb -s "$SERIAL" shell uiautomator dump /sdcard/photomath_result.xml
adb -s "$SERIAL" pull /sdcard/photomath_result.xml /tmp/photomath_result.xml
rg -n 'SOLVING STEPS|Multiply|Solve|84|x =|card_title|card_header' /tmp/photomath_result.xml
```

Observed crop behavior:

```text
Photomath may place the crop box too far right, cutting off the leftmost digit.
For example, 2x + 3 = 11 was read as x + 3 = 11 until the image/crop was
adjusted. For real-device experiments, always confirm the full expression is
inside the white crop coverage before tapping Solve.
```

Expensify:

```text
Package: org.me.mobiexpensifyg
Install type: split APK set from target-apps/expensify/_split
Expected flow: launch -> sign in/session -> Expenses -> add/scan receipt -> picker
Adapter status: locators still need real-device verification
Current real-device state: installed successfully; launches to sign-in screen.
Useful real-device locators observed:
- resource-id="username" class="android.widget.EditText"
- text="Phone or email"
- content-desc="Continue"
- content-desc="Sign in with Google"
Blocker: account/session setup is required before receipt scanning can be tested.
```

Expensify install and first-run check:

```bash
adb -s "$SERIAL" install-multiple -r -t \
  target-apps/expensify/_split/base.apk \
  target-apps/expensify/_split/split_config.arm64_v8a.apk \
  target-apps/expensify/_split/split_config.xxhdpi.apk

adb -s "$SERIAL" shell monkey -p org.me.mobiexpensifyg -c android.intent.category.LAUNCHER 1
adb -s "$SERIAL" shell uiautomator dump /sdcard/expensify.xml
adb -s "$SERIAL" pull /sdcard/expensify.xml /tmp/expensify.xml
rg -n 'username|Phone or email|Continue|Sign in|Expenses|Scan receipt' /tmp/expensify.xml
```

Noom:

```text
Package: com.wsl.noom
Install type: split APK set from target-apps/noom/_split
Required splits on Pixel-class arm64/xxhdpi real device:
- base.apk
- split_config.arm64_v8a.apk
- split_config.xxhdpi.apk
- split_config.en.apk
- split_preloadedDb_prod.apk
Current real-device state: installed successfully; launches to onboarding.
Useful real-device text observed:
- "Get started"
- "I already have an account"
Blocker: account/subscription/onboarding is required before meal/photo logging
can be tested.
```

Noom extraction, install, and first-run check:

```bash
mkdir -p target-apps/noom/_split
unzip -o \
  target-apps/noom/com.wsl.noom_14.12.0-302330_4arch_7dpi_4lang_1feat_7e7c7a76eac0cfac9835f3b156433d26_apkmirror.com.apkm \
  base.apk \
  split_config.arm64_v8a.apk \
  split_config.xxhdpi.apk \
  split_config.en.apk \
  split_preloadedDb_prod.apk \
  -d target-apps/noom/_split

adb -s "$SERIAL" install-multiple -r -t \
  target-apps/noom/_split/base.apk \
  target-apps/noom/_split/split_config.arm64_v8a.apk \
  target-apps/noom/_split/split_config.xxhdpi.apk \
  target-apps/noom/_split/split_config.en.apk \
  target-apps/noom/_split/split_preloadedDb_prod.apk

adb -s "$SERIAL" shell monkey -p com.wsl.noom -c android.intent.category.LAUNCHER 1
adb -s "$SERIAL" shell uiautomator dump /sdcard/noom.xml
adb -s "$SERIAL" pull /sdcard/noom.xml /tmp/noom.xml
rg -n 'Get started|I already have an account|Log|Meals|Add food' /tmp/noom.xml
```

### Troubleshooting

Restart adb if the server is wedged:

```bash
adb kill-server
adb start-server
adb devices
```

If `uiautomator dump` hangs or reports `ERROR: could not get idle state`, take
a screenshot and use coordinate taps; dialogs or animations often prevent idle.

If a screenshot command with host redirection fails in a sandbox:

```bash
adb -s "$SERIAL" shell screencap -p /sdcard/screen.png
adb -s "$SERIAL" pull /sdcard/screen.png /tmp/screen.png
```

If an APKMirror `.apkm` file will not install directly, extract it and install
matching split APKs with `install-multiple`. The Verify helper in
`verify/backend/drivers/emulator_manager.py` automates this for adapters.
