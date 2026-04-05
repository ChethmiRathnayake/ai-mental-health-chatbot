import React, { useState } from "react";
import { View, Text, Button } from "react-native";
import { recordAndExtract } from "../native/VoiceFeatures";
import { ingest, baselineStatus } from "../api";

export default function VoiceTask({ route, navigation }: any) {
  const prompt: string = route.params.prompt;
  const nextScreen: string = route.params.nextScreen;
  const [status, setStatus] = useState("");

  async function onRecord() {
    setStatus("Recording...");
    const voice = await recordAndExtract(3500);

    setStatus("Submitting voice baseline timestep...");
    const result = await ingest({
      text: "", // voice-only step allowed (text empty)
      keystrokes: [],
      voice_features: voice,
    });

    const bs = await baselineStatus();
    setStatus(`Saved. baseline samples=${bs.n_samples}, ready=${bs.is_ready}. PredReady=${result.ready}`);
    navigation.navigate(nextScreen);
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 18, fontWeight: "700" }}>Voice Task</Text>
      <Text style={{ marginTop: 12 }}>{prompt}</Text>

      <View style={{ marginTop: 12 }}>
        <Button title="Record 3.5s and Submit" onPress={onRecord} />
      </View>

      {!!status && <Text style={{ marginTop: 12 }}>{status}</Text>}
    </View>
  );
}