import AsyncStorage from "@react-native-async-storage/async-storage";
import { api, setAuthToken } from "./api";

const KEY = "access_token";

export async function signup(email: string, password: string) {
  const { data } = await api.post("/auth/signup", { email, password });
  await AsyncStorage.setItem(KEY, data.access_token);
  setAuthToken(data.access_token);
  return data.access_token as string;
}

export async function signinOAuth(email: string, password: string) {
  // Swagger OAuth2PasswordRequestForm expects form-encoded fields:
  const form = new URLSearchParams();
  form.append("username", email);
  form.append("password", password);

  const { data } = await api.post("/auth/token", form.toString(), {
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });

  await AsyncStorage.setItem(KEY, data.access_token);
  setAuthToken(data.access_token);
  return data.access_token as string;
}

export async function loadToken() {
  const token = await AsyncStorage.getItem(KEY);
  setAuthToken(token);
  return token;
}

export async function signout() {
  await AsyncStorage.removeItem(KEY);
  setAuthToken(null);
}