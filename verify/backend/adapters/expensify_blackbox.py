"""
Adapter stub for Expensify — closed-source business expense management with
SmartScan receipt OCR. Direct analogue to budget-lens but commercial backend.

Modality: image (SROIE2019 receipt images drop into "scan from library").

Status: scaffold only. config is populated, but `_drive_app` raises
NotImplementedError until the locators are verified against a real
`uiautomator2.dump_hierarchy()` taken on the provisioned AVD. See
analysis/verify_report_blackbox.md §15.5 for the verification recipe.

Per the README in target-apps/expensify/, the UI flow is:
  launch → login (snapshot must contain saved session) → bottom tab "Expenses"
  → "+" FAB → "Scan receipt" → "From library" → file picker → wait for
  "SmartScan in progress" → read merchant / amount / date.

Pinning: Expect TLS pinning in the React-Native build to require apk-mitm
*and possibly* a Frida pass before the NetworkObserver sees plaintext.
"""
from __future__ import annotations

from typing import Any, Dict

from verify.backend.adapters.blackbox_base import BlackBoxAdapter, BlackBoxConfig
from verify.backend.observers import DEFAULT_LLM_HOSTS


_EXPENSIFY_HOSTS = DEFAULT_LLM_HOSTS + (
    "*.expensify.com",
    "*.expensify.net",
)


class ExpensifyAdapter(BlackBoxAdapter):
    name = "expensify"
    supported_modalities = ["image"]

    config = BlackBoxConfig(
        package_name="org.me.mobiexpensifyg",
        main_activity="",
        apkm_filename="org.me.mobiexpensifyg_9.3.58-9-509035809_4arch_7dpi_9773f02585b3e805bfe6177715f2b69c_apkmirror.com.apkm",
        avd_name="verify_pixel7",
        snapshot_name="clean",
        llm_hosts=_EXPENSIFY_HOSTS,
        primary_backend_host="*.expensify.com",
        runtime_permissions=[
            "android.permission.READ_MEDIA_IMAGES",
            "android.permission.READ_EXTERNAL_STORAGE",
        ],
        timeout_s=180,  # SmartScan is asynchronous; allow time for the OCR job
    )

    def _drive_app(self, driver, input_item: Dict[str, Any]) -> str:
        # TODO(blackbox-phase2): verify locators against a real dump_hierarchy.
        #
        # Sketch of the expected flow once locators are confirmed:
        #   src = input_item.get("path")
        #   driver.push_image(src, "/sdcard/Download/receipt.jpg")
        #   driver.tap({"text": "Expenses"})
        #   driver.tap({"descriptionContains": "Add"})       # FAB
        #   driver.tap({"text": "Scan receipt"})
        #   driver.tap({"text": "From library"})
        #   driver.tap({"textContains": "receipt"})
        #   # SmartScan polls the server; wait for the edit fields to populate
        #   driver.wait_for({"resourceIdMatches": r".*:id/merchant.*"}, timeout=120)
        #   merchant = driver.read({"resourceIdMatches": r".*:id/merchant.*"})
        #   amount   = driver.read({"resourceIdMatches": r".*:id/amount.*"})
        #   date     = driver.read({"resourceIdMatches": r".*:id/date.*"})
        #   return f"Merchant: {merchant}\nAmount: {amount}\nDate: {date}"
        raise NotImplementedError(
            "ExpensifyAdapter._drive_app: locators not yet verified. "
            "See analysis/verify_report_blackbox.md §15.5."
        )
