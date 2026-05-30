# Security Audit — chexpert-pathology-classifier

## Version: 1.1.0 — Security Hardened
**Audit Date:** 2026-05-30
**Auditor:** Internal Security Review

## Summary
A full OWASP-Top-10 audit of this clinical AI FastAPI service identified one CRITICAL pickle-deserialization vulnerability and multiple HIGH-severity issues including completely unauthenticated inference endpoints, CORS wildcard exposure, missing security headers, and unsafe DICOM parsing. All CRITICAL and HIGH issues have been remediated and verified against the existing 182-test suite (all passing).

## Findings & Fixes

### CRITICAL

- **Insecure deserialization via `torch.load` without `weights_only=True` (CWE-502)**: `src/api.py` called `torch.load(checkpoint_path, map_location=device)` without restricting deserialization. A malicious actor with write access to the checkpoint path could embed arbitrary Python code in a pickle payload, achieving remote code execution on startup. Fixed by adding `weights_only=True` to the `torch.load` call in `src/api.py`.

### HIGH

- **No authentication on `/predict` and `/predict/batch`**: Both inference endpoints accepted any request with no credential check, exposing a clinical AI service — processing patient chest X-rays — to the public internet. Fixed by adding an `X-API-Key` header dependency (`_require_api_key`) to both routes. The key is read from the `API_KEY` environment variable; if unset, a cryptographically random key is generated per process (blocking all unauthenticated access). Two new tests enforce 401 rejection on missing or wrong keys.

- **No image payload size limit (memory DoS)**: No upper bound on the decoded base64 image size allowed an attacker to send arbitrarily large payloads (e.g., a 500 MB base64 string) to exhaust server memory. Fixed by enforcing a 20 MB cap in both the Pydantic validator (before base64 decoding, using the ~4/3 size ratio) and in `_decode_image` (after decoding, before PIL parsing).

- **Internal exception details leaked in API error responses**: `_decode_image` returned `detail=f"Invalid image data: {exc}"`, which could expose PIL class names, file paths, or library internals from the exception message. Fixed by catching all exceptions and returning a generic message; the original exception is logged at DEBUG level for diagnostics only.

- **CORS not configured — wildcard implicit**: No `CORSMiddleware` was added, meaning the framework defaults were in effect and browser-based callers from any origin could invoke the API. Fixed by adding `CORSMiddleware` with `allow_origins=ALLOWED_ORIGINS` defaulting to an empty list (deny all). Operators set the `CORS_ORIGINS` environment variable to explicitly permit trusted domains.

- **No security response headers**: No middleware set `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, or `Cache-Control` headers, leaving responses vulnerable to MIME sniffing, clickjacking, and browser caching of sensitive inference results. Fixed by adding an `add_security_headers` middleware that attaches all five headers to every response.

- **`load_dicom()` had no error handling — malformed DICOM crashes process**: `pydicom.dcmread(path)` was called with no try/except. A crafted DICOM file (e.g., with a malformed transfer syntax or corrupt compressed pixel stream) raises a variety of exceptions inside pydicom internals, which would propagate as an unhandled 500 or crash the worker. Fixed by wrapping `dcmread` in a try/except that re-raises a typed `ValueError` with a safe message; `FileNotFoundError` is allowed to propagate as-is.

- **`dicom_to_tensor()` had no error handling around pixel extraction and transform**: `dcm.pixel_array` decompresses image data (JPEG2000, RLE, etc.) and can raise parser-specific exceptions on malicious input. The downstream windowing and PIL/torchvision transforms can similarly crash. Fixed by wrapping `pixel_array` extraction in its own try/except (to catch decompression attacks specifically) and the windowing + transform pipeline in a second try/except, both re-raising as `ValueError`.

- **`mlflow.db` committed to the repository**: The SQLite experiment-tracking database was tracked by git (`git ls-files` confirmed it). It contains training metadata, hyperparameters, artifact paths, and run history, which could expose environment details to anyone with repository access. Fixed by adding `mlflow.db` to `.gitignore` and removing it from the git index (`git rm --cached mlflow.db`).

- **Under-pinned `python-multipart` (CVE-2024-53498)**: `requirements.txt` specified `python-multipart>=0.0.6`, which includes versions vulnerable to a ReDoS in the multipart boundary parser. Updated minimum to `>=0.0.18`.

- **Under-pinned `Pillow` (security fixes in 10.3.0+)**: `requirements.txt` specified `Pillow>=10.0.0`, which includes versions with known image-parsing vulnerabilities (CVE-2024-28219 and related). Updated minimum to `>=10.3.0`.

### MEDIUM

- **Path traversal in `CheXpertDataset.__getitem__` via malicious CSV**: The expression `self.data_root / row["Path"]` could resolve outside `data_root` if a CSV row contained a value like `../../etc/passwd`. This is a training-time path (not the API), but the dataset could be initialized with an untrusted CSV. Fixed in `src/data.py` by resolving the candidate path and checking it starts with the resolved `data_root` before opening.

- **No rate limiting on inference endpoints**: The API has no per-IP or per-key throttle. An attacker could run sustained inference load to exhaust GPU/CPU resources. Recommendation: add `slowapi` or an upstream reverse-proxy rate limit (e.g., nginx `limit_req`). Not remediated in code as it requires infrastructure configuration, but the authentication layer now provides a prerequisite identity anchor for per-key limiting.

- **`base64.b64decode` called without `validate=True`**: The original decode silently ignored non-base64 characters. Changed to `base64.b64decode(image_b64, validate=True)` so malformed strings raise immediately rather than producing silently corrupted bytes fed into PIL.

### LOW

- **`BatchPredictRequest.images_b64` had no `min_length` constraint**: An empty list bypassed Pydantic and required a manual check in the route handler. Fixed by adding `min_length=1` to the `Field` definition, making the constraint declarative and removing the redundant in-route check.

- **`PIL Image.verify()` not called before decode**: Images were opened and converted without first verifying they are well-formed. Added `image.verify()` before the final `.convert("RGB")` call to catch truncated or corrupted files early. Note: `verify()` consumes the stream, so the image is re-opened from the original bytes after verification.

## Status
All CRITICAL and HIGH issues resolved. Full test suite: 182 tests passing (0 failures).
