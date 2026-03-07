import React, { useState } from "react";
import { View, TextInput, Button, Text } from "react-native";
import { signinOAuth, signup } from "../auth";

export default function LoginScreen({ onAuthed }: { onAuthed: () => void }) {
  const [email, setEmail] = useState("test@test.com");
  const [password, setPassword] = useState("123456");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function doSignin() {
    setErr(null); setBusy(true);
    try { await signinOAuth(email, password); onAuthed(); }
    catch (e: any) { setErr(e?.response?.data?.detail ?? e.message); }
    finally { setBusy(false); }
  }

  async function doSignup() {
    setErr(null); setBusy(true);
    try { await signup(email, password); onAuthed(); }
    catch (e: any) { setErr(e?.response?.data?.detail ?? e.message); }
    finally { setBusy(false); }
  }

  return (
    <View style={{ flex: 1, padding: 20, justifyContent: "center", gap: 12 }}>
      <Text style={{ fontSize: 22, fontWeight: "600" }}>Login</Text>
      <TextInput value={email} onChangeText={setEmail} placeholder="email"
        autoCapitalize="none" style={{ borderWidth: 1, padding: 12, borderRadius: 8 }} />
      <TextInput value={password} onChangeText={setPassword} placeholder="password"
        secureTextEntry style={{ borderWidth: 1, padding: 12, borderRadius: 8 }} />
      {err ? <Text style={{ color: "red" }}>{String(err)}</Text> : null}
      <Button title={busy ? "..." : "Sign in"} onPress={doSignin} disabled={busy} />
      <Button title={busy ? "..." : "Sign up"} onPress={doSignup} disabled={busy} />
    </View>
  );
}