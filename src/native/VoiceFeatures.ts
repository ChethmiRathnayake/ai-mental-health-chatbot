import { NativeModules } from "react-native";
const { VoiceFeatureExtractor } = NativeModules;

export type VoiceFeatures = {
  pitch_variance: number;
  volume_fluctuation: number;
  tone_variability: number;
};

// records audio for durationMs and returns features
export async function recordAndExtract(durationMs: number): Promise<VoiceFeatures> {
  return await VoiceFeatureExtractor.recordAndExtract(durationMs);
}