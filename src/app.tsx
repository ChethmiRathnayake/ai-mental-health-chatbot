import React, { useEffect, useState } from "react";
import { View, ActivityIndicator } from "react-native";
import LoginScreen from "./screens/LoginScreen";
import ChatScreen from "./screens/ChatScreen";
import { loadToken } from "./auth";

export default function App() {
  const [ready, setReady] = useState(false);
  const [authed, setAuthed] = useState(false);

  useEffect(() => {
    (async () => {
      const token = await loadToken();
      setAuthed(!!token);
      setReady(true);
    })();
  }, []);

  if (!ready) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <ActivityIndicator />
      </View>
    );
  }

  return authed ? <ChatScreen /> : <LoginScreen onAuthed={() => setAuthed(true)} />;
}