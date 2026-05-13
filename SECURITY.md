# Security Policy

`node-rapids` (the `@rapidsai/*` family of npm packages, this repository) is
a collection of Node.js native addons that bridge JavaScript / TypeScript
to RAPIDS C++/CUDA libraries (cuDF, cuML, cuGraph, cuSpatial, RMM, etc.),
plus thin bindings to CUDA, GLFW, WebGL, and a small set of file-format
readers. It is invoked in-process inside Node.js or Electron, and inherits
the caller's privilege.

The security posture is shaped by three boundaries unique to this project:

1. The **JavaScript ↔ N-API ↔ C++ boundary** — where JavaScript values
   become C++ types, including pointers and sizes consumed by CUDA APIs.
2. The **file-format readers** in `@rapidsai/io` — currently a LAS point-
   cloud parser, with the same parser-of-untrusted-bytes risk class as
   any other format reader.
3. The **install / distribution pipeline** — pre-built native modules are
   downloaded at install time.

## Reporting a Vulnerability

Please report security vulnerabilities privately through one of the channels
below. **Do not open a public GitHub issue, PR, or discussion** for a
suspected vulnerability.

1. **NVIDIA Vulnerability Disclosure Program (preferred)**
   <https://www.nvidia.com/en-us/security/>
   Submit through the NVIDIA PSIRT web form. This is the fastest path to
   triage and tracking.

2. **Email NVIDIA PSIRT**
   psirt@nvidia.com — encrypt sensitive reports with the
   [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key).

3. **GitHub Private Vulnerability Reporting**
   Use the **Security** tab on this repository → *Report a vulnerability*.

Please include, where possible:

- Affected package (e.g. `@rapidsai/cudf`, `@rapidsai/io`,
  `@rapidsai/core`) and version
- Affected component (a specific binding, the LAS parser, the install
  script, a demo server)
- Node / Electron version, CUDA version, GPU model, and OS
- Reproduction steps and a minimal proof-of-concept input
- Impact assessment (memory corruption, code execution, DoS, info
  disclosure)
- Any relevant CWE / CVE identifiers

NVIDIA PSIRT will acknowledge receipt and coordinate triage, fix
development, and coordinated disclosure. More on NVIDIA's response
process: <https://www.nvidia.com/en-us/security/psirt-policies/>.

## Security Architecture & Context

**Classification:** Library — a set of Node.js native addons (N-API via
`node-addon-api`) wrapping RAPIDS C++/CUDA libraries, distributed as npm
packages. Pre-built binaries are also delivered via container images
under `ghcr.io/rapidsai/node`.

**Primary security responsibility:** Translate between JavaScript values
and the underlying C++/CUDA APIs safely — converting numbers, typed
arrays, and buffers into well-typed C++ pointers, sizes, and CUDA-stream-
appropriate operations without exposing the host process to memory
corruption or arbitrary memory access. Plus, where the bindings include
their own parsers (LAS in `@rapidsai/io`), parse untrusted bytes
defensively.

**Components and trust boundaries:**

- **`modules/core/`** — `@rapidsai/core`. The shared N-API addon
  scaffolding all other modules use.
- **`modules/cuda/`** — `@rapidsai/cuda`. Direct bindings to CUDA
  Runtime APIs (allocations, memcpy, memset, stream/event handles).
- **`modules/rmm/`** — `@rapidsai/rmm`. RAPIDS Memory Manager bindings.
- **`modules/cudf/`, `cuml/`, `cugraph/`, `cuspatial/`** — bindings to
  the respective RAPIDS libraries. Each carries the upstream library's
  attack surface (file-format readers in cudf, model imports in cuml,
  graph topology in cugraph, geometric parsers in cuspatial) through
  this layer.
- **`modules/io/`** — `@rapidsai/io`. File-format I/O including a CUDA
  LAS (LiDAR / point cloud) parser at `modules/io/src/las.cu`.
- **`modules/glfw/`, `modules/webgl/`, `modules/deck.gl/`** — windowing
  and rendering bindings (GLFW, OpenGL ES / WebGL, deck.gl).
- **`modules/jsdom/`** — a jsdom integration used to host browser-style
  rendering in Node.
- **`modules/demo/`** — example applications. Notably includes
  network-listening demo servers (`demo/api-server/`, `demo/viz-app/`,
  `demo/client-server/`, `demo/ssr/`). These are demonstrations, not
  hardened services, and should not be treated as production templates.
- **`modules/core/bin/rapidsai_install_native_module.js`** — install-time
  script that downloads a pre-built native module tarball over HTTPS
  matching the host's CUDA driver, GPU compute capability, and Linux
  distribution.

**Out of scope for this policy:** vulnerabilities in CUDA, the NVIDIA
driver, the upstream RAPIDS libraries themselves (report those via the
respective `rapidsai/*` repositories), Node.js, Electron, V8, GLFW, jsdom,
or deck.gl. Vulnerabilities in the *bindings* — how this repo marshals
values across the JS ↔ C++ boundary, how it handles errors, how it
constructs CUDA API arguments — are in scope, as are bugs in the LAS
parser shipped here.

## Threat Model

The threats below trace to specific components and patterns in this
repository. Several have already been observed and remediated through
the [RAPIDS Security Audit](https://github.com/orgs/rapidsai/projects/207);
they are listed so callers and integrators understand the classes of bugs
the library defends against.

1. **Arbitrary memory access via JavaScript-controlled pointer casts.**
   N-API bindings throughout `modules/` accept JavaScript numbers and
   typed-array buffers and convert them into C++ pointers (e.g.
   `void*`, typed `T*`). When the conversion does not validate that
   the resulting pointer is owned, live, and type-correct, hostile
   JavaScript can pass an arbitrary integer and trigger reads or
   writes to arbitrary process memory. This is a class-of-bug spanning
   many of the bindings, not a single call site.

2. **Unchecked sizes feeding `cudaMemcpy` / `cudaMemset`.**
   Several `@rapidsai/cuda` and dependent bindings pass JavaScript-
   supplied lengths directly to `cudaMemcpy` and `cudaMemset`. Sizes
   that exceed the destination allocation produce out-of-bounds GPU
   memory access — readable as info disclosure or writable as
   corruption — without raising a host-side fault.

3. **LAS point-cloud parser memory safety.**
   `modules/io/src/las.cu` parses LAS headers and point records from
   caller-supplied bytes. Integer overflow in header arithmetic and
   out-of-bounds access in point-record decoding are known issues
   reachable from a hostile `.las` file. Header parsing additionally
   leaks both host and GPU memory on certain error paths, providing a
   long-running DoS vector for callers that retry on parse failures.

4. **Command injection in a former SQL module (historical).**
   An earlier `@rapidsai/sql` module executed Python via shell
   interpolation, allowing JavaScript-controlled input to inject
   commands. The module has been removed from the repository; the
   class of bug is mentioned here because consumers pinning to old
   versions of the package are still exposed.

5. **Native-module download without integrity verification.**
   `modules/core/bin/rapidsai_install_native_module.js` downloads a
   pre-built `.tar.gz` over HTTPS at install time, matched by CUDA
   driver / GPU compute capability / OS. The script relies on TLS for
   integrity; there is no checksum or signature check against a
   trusted manifest. A compromised download host, a hijacked release
   asset, or an in-path TLS interception on the install machine
   yields arbitrary code execution at npm-install time.

6. **GitHub Actions template injection.**
   The repository's CI workflows historically interpolated `${{ ... }}`
   expressions derived from PR metadata into shell `run:` blocks,
   yielding arbitrary command execution in the runner with the
   workflow's secret scope. The audit produced fixes; the risk class
   recurs on new workflow contributions.

7. **Demo servers are not hardened.**
   `modules/demo/api-server`, `demo/viz-app`, `demo/client-server`,
   and `demo/ssr/*` are network-listening Node.js applications shipped
   as examples. They do not implement authentication, authorization,
   or input validation appropriate for production. Using them as
   production templates re-creates whatever vulnerabilities they
   contain in the deployer's stack.

8. **Underlying-library attack surface flows through the binding.**
   `@rapidsai/cudf` exposes the same parser-of-untrusted-bytes risk
   class as upstream `cudf` (Parquet / ORC / JSON / CSV / Avro);
   `@rapidsai/cuml` exposes the FIL Treelite-import surface;
   `@rapidsai/cugraph` exposes type-confusion risks at the C ABI
   boundary; `@rapidsai/cuspatial` exposes geometric-parser surface.
   See the upstream `rapidsai/cudf`, `cuml`, `cugraph`, and
   `cuspatial` SECURITY.md files for those threat models — this
   repo's bindings are an additional propagation path, not a
   replacement.

## Critical Security Assumptions

node-rapids is a library and inherits the caller's privilege; the
following are assumed of the caller / deployer.

- **JavaScript callers respect binding type contracts.**
  N-API bindings trust the JavaScript-side type and value. A caller
  that passes raw numbers where typed buffer handles are expected can
  drive arbitrary dereferences in the C++ layer. Treat the node-rapids
  API as an in-process FFI, not a sandbox; do not expose the bindings
  directly to untrusted JavaScript (browser-side scripts in an
  Electron renderer, third-party plugins, REPL-style code execution
  from network input).

- **Caller-supplied sizes are bounded.**
  Bindings that forward sizes to `cudaMemcpy`, `cudaMemset`, or
  allocation primitives assume the caller has bounded those values
  against the relevant buffer. Until per-binding bounds checks are
  comprehensive, callers should validate sizes upstream.

- **Inputs to the LAS parser may be hostile, but the caller decides
  whether to trust them.**
  Parsing LAS files from external sources should be done in a process
  with the smallest viable blast radius (separate worker, container,
  memory limits). The known integer-overflow and OOB issues in the
  parser bound what the binding can promise.

- **Install-time downloads are verified out-of-band, or pinned.**
  Until the install script enforces a checksum or signature against a
  trusted manifest, the integrity of `npm install @rapidsai/*` rests
  on TLS to the download host. Operators concerned with build-host
  integrity should either pre-fetch and locally pin the native module
  tarball (using `RAPIDSAI_SKIP_DOWNLOAD=1`) or run installs in a
  network environment they trust.

- **Demo servers are not production templates.**
  The applications under `modules/demo/` are illustrative. Lifting
  any of them into a production deployment re-creates whatever
  network exposure they document — typically unauthenticated, on
  permissive CORS, on `0.0.0.0`.

- **Workflows that use this repo enforce least privilege.**
  The CI workflows have a history of template-injection issues and
  the `secrets: inherit` / mutable-ref findings that recur across the
  RAPIDS organization. New workflow contributions should be reviewed
  with that history in mind.

- **GPU memory is not a confidentiality boundary.**
  Multiple processes sharing a GPU may observe each other's GPU
  memory through driver-level side channels. node-rapids assumes the
  caller has provisioned the GPU appropriately (MIG, exclusive
  process, container isolation) when confidentiality matters.

- **The upstream RAPIDS trust model applies.**
  Calling `@rapidsai/cudf.read_parquet`, an `@rapidsai/cuml` Treelite
  importer, or any other binding that forwards into a RAPIDS library
  carries that library's threat model with it. Consult the upstream
  SECURITY.md when integrating against untrusted inputs.

## Supported Versions

The published `@rapidsai/*` packages track the RAPIDS release cadence.
Older versions are not back-ported with security fixes; upgrade to the
latest published version to receive fixes — and to pick up upstream
RAPIDS-library updates.

## Dependency Security

node-rapids depends on Node.js, the `node-addon-api` runtime, the
upstream RAPIDS native libraries (libcudf, libcuml, libcugraph,
libcuspatial, librmm), CUDA, GLFW, jsdom, and the deck.gl /
loaders.gl ecosystem. CVE-driven updates to any of these may require
rebuilding the native modules. Operators ingesting untrusted inputs
should track upstream advisories and avoid long-pinned node-rapids
builds.
