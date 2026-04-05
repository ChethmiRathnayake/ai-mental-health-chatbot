package com.cogloadapp

import android.media.*
import com.facebook.react.bridge.*
import org.jtransforms.fft.DoubleFFT_1D
import kotlin.math.*

class VoiceFeatureExtractorModule(reactContext: ReactApplicationContext) :
  ReactContextBaseJavaModule(reactContext) {

  override fun getName(): String = "VoiceFeatureExtractor"

  @ReactMethod
  fun recordAndExtract(durationMs: Int, promise: Promise) {
    try {
      val sampleRate = 16000
      val channelConfig = AudioFormat.CHANNEL_IN_MONO
      val audioFormat = AudioFormat.ENCODING_PCM_16BIT
      val minBuf = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
      val recorder = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
        sampleRate, channelConfig, audioFormat, minBuf)

      val totalSamples = (durationMs * sampleRate) / 1000
      val buffer = ShortArray(minBuf / 2)
      val samples = DoubleArray(totalSamples)
      var idx = 0

      recorder.startRecording()
      while (idx < totalSamples) {
        val read = recorder.read(buffer, 0, buffer.size)
        for (i in 0 until read) {
          if (idx >= totalSamples) break
          samples[idx] = buffer[i].toDouble() / 32768.0
          idx++
        }
      }
      recorder.stop()
      recorder.release()

      val feats = extractFeatures(samples, sampleRate)

      val map = Arguments.createMap()
      map.putDouble("pitch_variance", feats.pitchVar)
      map.putDouble("volume_fluctuation", feats.rmsVar)
      map.putDouble("tone_variability", feats.centroidVar)
      promise.resolve(map)

    } catch (e: Exception) {
      promise.reject("VOICE_FEATURE_ERROR", e)
    }
  }

  data class Feats(val pitchVar: Double, val rmsVar: Double, val centroidVar: Double)

  private fun extractFeatures(x: DoubleArray, sr: Int): Feats {
    val frame = 1024
    val hop = 512
    val rms = mutableListOf<Double>()
    val pitch = mutableListOf<Double>()
    val centroid = mutableListOf<Double>()

    var i = 0
    while (i + frame <= x.size) {
      val frameX = x.copyOfRange(i, i + frame)
      rms.add(calcRMS(frameX))
      pitch.add(estimatePitch(frameX, sr))
      centroid.add(spectralCentroid(frameX, sr))
      i += hop
    }

    fun variance(v: List<Double>): Double {
      val clean = v.filter { it.isFinite() && it > 0.0 }
      if (clean.size < 2) return 0.0
      val m = clean.sum() / clean.size
      return clean.sumOf { (it - m) * (it - m) } / (clean.size - 1)
    }

    return Feats(
      pitchVar = variance(pitch),
      rmsVar = variance(rms),
      centroidVar = variance(centroid)
    )
  }

  private fun calcRMS(x: DoubleArray): Double {
    val m = x.sumOf { it * it } / x.size
    return sqrt(m)
  }

  // crude pitch estimate (autocorrelation peak) – MVP
  private fun estimatePitch(x: DoubleArray, sr: Int): Double {
    val minF = 60.0
    val maxF = 400.0
    val minLag = (sr / maxF).toInt()
    val maxLag = (sr / minF).toInt()

    var bestLag = -1
    var best = 0.0

    for (lag in minLag..maxLag) {
      var sum = 0.0
      for (i in 0 until (x.size - lag)) {
        sum += x[i] * x[i + lag]
      }
      if (sum > best) {
        best = sum
        bestLag = lag
      }
    }
    return if (bestLag > 0) sr.toDouble() / bestLag else 0.0
  }

  private fun spectralCentroid(x: DoubleArray, sr: Int): Double {
    val n = x.size
    val fft = DoubleFFT_1D(n.toLong())
    val buf = DoubleArray(n * 2)
    for (i in 0 until n) {
      buf[2*i] = x[i]
      buf[2*i + 1] = 0.0
    }
    fft.complexForward(buf)

    var num = 0.0
    var den = 1e-12
    val half = n / 2
    for (k in 1 until half) {
      val re = buf[2*k]
      val im = buf[2*k + 1]
      val mag = sqrt(re*re + im*im)
      val freq = k.toDouble() * sr / n
      num += freq * mag
      den += mag
    }
    return num / den
  }
}