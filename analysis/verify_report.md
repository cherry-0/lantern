# Verify App Adapter Analysis Report

This report details how the adapters in the `verify` application bridge the gap between the native AI pipelines of various target apps and the verification framework. It also covers the fallback mechanism that uses OpenRouter.

## 1. Adapter Architecture

The `verify` app uses a plugin-style adapter architecture to interact with the diverse set of target applications. Each target app has a corresponding adapter that inherits from `verify.backend.adapters.base.BaseAdapter`. This base class defines a standardized interface that the Verify orchestrator uses to run and evaluate each app's AI pipeline.

### Core `BaseAdapter` Interface:

-   **`check_availability()`**: This method is crucial. It checks if the adapter can run, either by locating the native pipeline's dependencies or by confirming that a fallback mechanism (like an OpenRouter API key) is available.
-   **`run_pipeline(input_item)`**: This is the main execution entry point. The orchestrator calls this method with a normalized input (e.g., text, image). The adapter is responsible for translating this input into a format the target app's pipeline understands, running the pipeline, and returning a normalized `AdapterResult`.

## 2. Linking Native Pipelines via Environment Variable Injection

The primary strategy for linking to a native pipeline has shifted. Instead of just checking for the existence of a native library and then using a separate fallback, the new architecture **prioritizes running the *native* pipeline by redirecting its API calls to OpenRouter**.

This is achieved through a helper method in `BaseAdapter`:

-   **`_inject_openrouter_env()`**: This static method injects the user's `OPENROUTER_API_KEY` into several standard and app-specific environment variables at runtime (`OPENAI_API_KEY`, `LLM_API_KEY`, `VLM_API_KEY`, etc.).

### How It Works: `deeptutor` Example

The `deeptutor` adapter (`deeptutor.py`) is the prime example of this new pattern:

1.  **Environment Injection**: Before any `deeptutor` modules are imported, the adapter calls `_inject_deeptutor_env()`, which sets environment variables like `LLM_BINDING`, `LLM_HOST`, and `LLM_API_KEY`.
2.  **Native Import**: `deeptutor`'s internal services, like its `LLMService` which uses `litellm`, read these environment variables upon initialization. Because the variables point to `openrouter.ai`, `litellm` automatically routes all "OpenAI" or "GPT" model calls to OpenRouter instead of the actual OpenAI API.
3.  **Native Execution**: The adapter then directly invokes `deeptutor.runtime.orchestrator.ChatOrchestrator`. This is the *actual* native code path for the app's `chat` capability.
4.  **Asynchronous Handling**: Since the orchestrator is `async`, the adapter uses a `_run_async` helper to execute the coroutine in a separate thread, preventing conflicts with Verify's own event loop (e.g., when run via Streamlit).

This approach is powerful because it allows `verify` to test the *entire native application logic*—including its specific prompt construction, RAG pipeline, and context management—while still controlling the final LLM endpoint. The "pipeline" being run is truly native; only the final API call is rerouted.

## 3. OpenRouter Fallback Mechanism

A direct fallback to OpenRouter is still used, but its role has changed. It is now reserved for two primary scenarios:

1.  **When the native app cannot be imported at all** (e.g., it's a TypeScript/Electron app like `clone`).
2.  **When the native pipeline fails during execution** (e.g., `deeptutor`'s orchestrator fails to produce content).

### `clone` Adapter: The Pure Fallback Case

The `clone` adapter (`clone.py`) cannot import a Python-based pipeline. Its `check_availability()` method confirms this and verifies that an OpenRouter key is present. Its `run_pipeline()` method then directly calls `_run_on_frames`, which constructs a vision prompt and sends it to OpenRouter. This is consistent with the old architecture and remains the correct approach for non-Python pipelines.

### `deeptutor` Adapter: The Graceful Fallback

If `deeptutor`'s `_run_native()` method fails to produce any text content from the orchestrator's event stream, it gracefully falls back to calling `_run_openrouter_fallback()`. This fallback method, like in the old architecture, simulates the *spirit* of the native pipeline by using a specific tutor-persona prompt to ensure the output is contextually relevant for privacy evaluation.

## 4. Conclusion

The `verify` app's updated adapter architecture is more sophisticated and robust. By prioritizing **running the native pipeline with injected, rerouted credentials**, it achieves a higher-fidelity simulation of the target app's behavior. The `deeptutor` adapter's implementation is a clear demonstration of this advanced technique. The direct OpenRouter fallback is retained as a crucial safety net for non-importable apps or unexpected runtime failures. This two-tiered strategy—preferring native execution via environment routing, with a direct API call as a fallback—makes the verification process more accurate and resilient.
