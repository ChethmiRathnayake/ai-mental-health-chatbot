import React, { useMemo, useRef, useState } from "react";
import { View, Text, TextInput, Button } from "react-native";
import { createTypingTelemetry } from "../typingTelemetry";
import { recordAndExtract } from "../native/VoiceFeatures";
import { ingest, baselineStatus } from "../api";

export default function TypingTask({ route, navigation }: any) {
  const prompt: string = route.params.prompt;

  const [text, setText] = useState("");
  const [prev, setPrev] = useState("");
  const [status, setStatus] = useState("");

  const telem = useMemo(() => createTypingTelemetry(), []);
  const started = useRef(false);

  function onChange(next: string) {
    if (!started.current) {
      telem.start();
      started.current = true;
    }
    if (next.length < prev.length) telem.markBackspace();
    else telem.markChar();

    setPrev(next);
    setText(next);
  }

  async function onSubmit() {
    setStatus("Recording voice (optional baseline)...");

    // for baseline we can also do a quick voice sample
    const voice = await recordAndExtract(2500);

    const keystrokes = telem.stop();
    started.current = false;

    setStatus("Submitting baseline timestep...");
    const result = await ingest({
      text,
      keystrokes,
      voice_features: voice,
    });

    const bs = await baselineStatus();
    setStatus(`Saved. baseline samples=${bs.n_samples}, ready=${bs.is_ready}. Last pred ready=${result.ready}`);

    // navigate onward
    navigation.navigate(route.params.nextScreen);
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 18, fontWeight: "700" }}>Typing Task</Text>
      <Text style={{ marginTop: 12 }}>{prompt}</Text>

      <TextInput
        value={text}
        onChangeText={onChange}
        placeholder="Type your response..."
        multiline
        style={{ borderWidth: 1, borderRadius: 10, padding: 12, marginTop: 12 }}
      />

      <View style={{ marginTop: 12 }}>
        <Button title="Submit" onPress={onSubmit} />
      </View>

      {!!status && <Text style={{ marginTop: 12 }}>{status}</Text>}
    </View>
  );
}