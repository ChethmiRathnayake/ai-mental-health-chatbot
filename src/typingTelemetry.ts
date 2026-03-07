export type KeyEvt = { ts_ms: number; type: string; key?: string | null };

export function createTypingTelemetry() {
  let events: KeyEvt[] = [];
  let started = false;

  function start() {
    events = [];
    started = true;
  }

  function stop() {
    started = false;
    return events;
  }

  function markChar() {
    if (!started) return;
    events.push({ ts_ms: Date.now(), type: "char", key: null });
  }

  function markBackspace() {
    if (!started) return;
    events.push({ ts_ms: Date.now(), type: "backspace", key: "Backspace" });
  }

  return { start, stop, markChar, markBackspace };
}