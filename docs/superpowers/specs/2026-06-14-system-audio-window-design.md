# System Audio Window Design

## Goal

Add a command-line launched macOS window mode for live captions.

The new tool captures only computer output audio, transcribes it in real time, and displays
the transcript in a window. It does not translate, does not use speaker diarization, and does
not capture microphone input.

## Entry Point

Add a new console script:

```bash
uv run livecaption-window
```

This entry point runs alongside the existing `livecaption` CLI. The current terminal renderer,
file writer, translation path, microphone mode, and `both` mode remain unchanged.

Supported first-version options:

- `--asr-model`
- `--asr-lang`
- `--audiotee`
- `--include-pid`

The window tool should choose fixed defaults:

- Source: system audio only.
- Translation: disabled.
- Diarization: disabled.
- File output: disabled.

## Architecture

Reuse the existing audio and ASR pipeline:

```text
SystemAudioSource -> AsrWorker -> WindowRenderer
```

The entry point resolves `audiotee`, builds a recognizer with `diarize=False`, starts one
`SystemAudioSource`, and starts one `AsrWorker`.

The new window renderer is responsible only for presentation. It receives the same callback
shape as the existing pipeline:

- partial events update the current in-progress caption
- final events append finalized transcript text to history

No translation objects should be constructed. No microphone source should be constructed.

## Window UI

Use Python's standard Tkinter stack for the first version. This avoids adding a new runtime
dependency and keeps the implementation close to the current Python application.

The window should contain:

- a compact status row for startup, listening, stopping, and errors
- a scrollable transcript area for finalized captions
- a current-caption area for the latest partial transcript

The UI must be updated from the Tk main thread. ASR callbacks can run on worker threads, so the
renderer should enqueue UI work and drain it through Tk's event loop.

The first version should favor readability over visual complexity: high-contrast text, modest
padding, and no speaker labels.

## Lifecycle

Starting:

1. Create the Tk window.
2. Resolve `audiotee`.
3. Load and warm up ASR/VAD.
4. Start the system audio source and ASR worker.
5. Show "Listening" once capture begins.

Stopping:

- Closing the window requests shutdown.
- Ctrl-C in the launching terminal also requests shutdown.
- Shutdown stops the source, joins the worker with the same generous timeout used by the CLI,
  and then closes the window.

If `audiotee` cannot be found or started, or the recognizer cannot load, show the error in the
window status area and exit cleanly after the user closes the window.

## Error Handling

Keep the existing `SystemAudioSource` behavior for stalls, restarts, and all-zero permission
warnings. The terminal may still receive stderr warnings from that source.

The window should also expose high-level state:

- loading ASR model
- listening
- stopping
- stopped
- fatal startup error
- worker error

ASR worker errors should stop the pipeline rather than leaving the window pretending to listen.

## Testing

Add tests for logic that can run without macOS audio permissions or model downloads:

- a small window-renderer adapter/queue unit test proving callbacks are marshaled into UI work
  in order
- entry-point construction tests for the fixed mode defaults, especially that translation and
  diarization are disabled

Smoke verification after implementation:

- `uvx ruff check .`
- existing lightweight smoke tests that do not require model downloads when feasible
- manual run on macOS after `audiotee` is built:

```bash
uv run livecaption-window
```

## Out Of Scope

- Translation
- Speaker diarization
- Microphone capture
- Dual-source capture
- Packaging as a double-clickable `.app`
- Saving transcript files from the window mode
- Advanced native macOS UI polish
