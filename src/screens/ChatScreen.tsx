import React, { useEffect, useState } from "react";
import { View, TextInput, Button, Text, FlatList } from "react-native";
import { api } from "../api";

type Msg = { role: "user" | "assistant"; text: string };

export default function ChatScreen() {
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      const { data } = await api.post("/chat/start");
      setSessionId(data.session_id);
    })();
  }, []);

  async function send() {
    if (!sessionId || !text.trim()) return;
    const userText = text.trim();
    setText("");
    setMsgs((m) => [...m, { role: "user", text: userText }]);
    setBusy(true);

    // Minimal keystrokes (you will replace later with real typing telemetry)
    const keystrokes = userText.split("").map((ch, i) => ({
      key: ch,
      ts_ms: i * 80,
      type: "keydown",
    }));

    const { data } = await api.post(`/chat/message?session_id=${sessionId}`, {
      text: userText,
      keystrokes,
    });

    setMsgs((m) => [...m, { role: "assistant", text: data.reply }]);
    setBusy(false);
  }

  return (
    <View style={{ flex: 1, padding: 12 }}>
      <FlatList
        data={msgs}
        keyExtractor={(_, i) => String(i)}
        renderItem={({ item }) => (
          <Text style={{ marginVertical: 6 }}>
            <Text style={{ fontWeight: "700" }}>{item.role}: </Text>
            {item.text}
          </Text>
        )}
      />
      <View style={{ flexDirection: "row", gap: 8 }}>
        <TextInput
          value={text}
          onChangeText={setText}
          placeholder="Type..."
          style={{ flex: 1, borderWidth: 1, padding: 10, borderRadius: 8 }}
        />
        <Button title={busy ? "..." : "Send"} onPress={send} disabled={busy || !sessionId} />
      </View>
    </View>
  );
}