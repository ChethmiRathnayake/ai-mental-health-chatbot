import axios from "axios";
import AsyncStorage from "@react-native-async-storage/async-storage";

const BASE_URL = "http://112.135.68.189:8000";

import { API_BASE } from "./config";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
});

export function setAuthToken(token: string | null) {
  if (token) api.defaults.headers.common.Authorization = `Bearer ${token}`;
  else delete api.defaults.headers.common.Authorization;
}

async function authHeader() {
  const token = await AsyncStorage.getItem("token");
  return { Authorization: `Bearer ${token}` };
}

export async function signUp(email: string, password: string) {
  const res = await axios.post(`${BASE_URL}/auth/signup`, { email, password });
  return res.data.access_token as string;
}

export async function signIn(email: string, password: string) {
  const res = await axios.post(`${BASE_URL}/auth/signin`, { email, password });
  return res.data.access_token as string;
}

export async function baselineStatus() {
  const headers = await authHeader();
  const res = await axios.get(`${BASE_URL}/baseline/status`, { headers });
  return res.data;
}

export async function ingest(payload: any) {
  const headers = await authHeader();
  const res = await axios.post(`${BASE_URL}/ingest`, payload, { headers });
  return res.data;
}