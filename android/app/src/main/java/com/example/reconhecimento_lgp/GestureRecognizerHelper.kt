package com.example.reconhecimento_lgp

import android.content.Context
import android.os.SystemClock
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

class GestureRecognizerHelper(
    private val context: Context,
    private val listener: GestureRecognizerListener
) {

    private lateinit var handLandmarker: HandLandmarker
    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>

    private val input = ByteBuffer.allocateDirect(NUM_FEATURES * Float.SIZE_BYTES).order(ByteOrder.nativeOrder())
    private lateinit var output: Array<FloatArray>
    private val recentScores = ArrayDeque<FloatArray>()
    private var framesWithoutHand = 0

    init {
        try {
            setupHandLandmarker()
            setupTFLite()
        } catch (e: Exception) {
            listener.onError("Initialization failed: ${e.message}")
            throw e
        }
    }

    private fun setupHandLandmarker() {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(HAND_LANDMARKER_ASSET)
            .build()
        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(this::onHandLandmarkerResult)
            .setErrorListener { error -> listener.onError(error.message ?: "Unknown error") }
            .build()
        handLandmarker = HandLandmarker.createFromOptions(context, options)
    }

    private fun setupTFLite() {
        tflite = Interpreter(loadModelFile())
        labels = loadLabels()
        output = Array(1) { FloatArray(labels.size) }
    }

    private fun loadModelFile(): MappedByteBuffer {
        context.assets.openFd(MODEL_ASSET).use { fd ->
            FileInputStream(fd.fileDescriptor).use { stream ->
                return stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    private fun loadLabels(): List<String> {
        val json = context.assets.open(LABELS_ASSET).bufferedReader().use { it.readText() }
        val type = object : TypeToken<Map<String, Int>>() {}.type
        val map: Map<String, Int> = Gson().fromJson(json, type)
        return map.entries.sortedBy { it.value }.map { it.key }
    }

    fun recognizeLiveStream(image: MPImage) {
        handLandmarker.detectAsync(image, SystemClock.uptimeMillis())
    }

    private fun onHandLandmarkerResult(result: HandLandmarkerResult, image: MPImage) {
        listener.onResults(result)

        val hand = result.landmarks().firstOrNull()
        if (hand == null) {
            // Tolera falhas curtas de deteção antes de esquecer o histórico.
            if (++framesWithoutHand > MAX_FRAMES_WITHOUT_HAND && recentScores.isNotEmpty()) {
                recentScores.clear()
                listener.onGestureRecognized(null)
            }
            return
        }
        framesWithoutHand = 0

        recentScores.addLast(classify(hand, image.width, image.height))
        if (recentScores.size > SMOOTHING_WINDOW) recentScores.removeFirst()
        if (recentScores.size < MIN_FRAMES) return

        val average = FloatArray(labels.size)
        recentScores.forEach { scores -> scores.forEachIndexed { i, s -> average[i] += s / recentScores.size } }
        val best = average.indices.maxBy { average[it] }

        listener.onGestureRecognized(
            if (average[best] >= MIN_CONFIDENCE) Recognition(labels[best], average[best]) else null
        )
    }

    private fun classify(hand: List<NormalizedLandmark>, width: Int, height: Int): FloatArray {
        input.rewind()
        normalizeLandmarks(hand, width, height).forEach { input.putFloat(it) }
        input.rewind()
        tflite.run(input, output)
        return output[0].copyOf()
    }

    fun close() {
        if (this::handLandmarker.isInitialized) handLandmarker.close()
        if (this::tflite.isInitialized) tflite.close()
    }

    data class Recognition(val label: String, val confidence: Float)

    interface GestureRecognizerListener {
        fun onError(error: String)
        fun onResults(result: HandLandmarkerResult)
        fun onGestureRecognized(recognition: Recognition?)
    }

    companion object {
        private const val HAND_LANDMARKER_ASSET = "hand_landmarker.task"
        private const val MODEL_ASSET = "modelo_gestos_lgp.tflite"
        private const val LABELS_ASSET = "classes.json"

        private const val NUM_LANDMARKS = 21
        private const val NUM_FEATURES = NUM_LANDMARKS * 3

        // Tem de coincidir com SMOOTHING_WINDOW em ml/treinar_modelo.py
        private const val SMOOTHING_WINDOW = 10
        private const val MIN_FRAMES = 5
        private const val MIN_CONFIDENCE = 0.6f
        private const val MAX_FRAMES_WITHOUT_HAND = 5

        /** Igual a `normalize_landmarks` em ml/landmarks.py. */
        fun normalizeLandmarks(hand: List<NormalizedLandmark>, width: Int, height: Int): FloatArray {
            val wrist = hand[0]
            val points = FloatArray(NUM_FEATURES)
            var scale = 0f
            hand.forEachIndexed { i, lm ->
                val x = (lm.x() - wrist.x()) * width
                val y = (lm.y() - wrist.y()) * height
                points[i * 3] = x
                points[i * 3 + 1] = y
                points[i * 3 + 2] = (lm.z() - wrist.z()) * width
                scale = maxOf(scale, sqrt(x * x + y * y))
            }
            scale = maxOf(scale, 1e-6f)
            for (i in points.indices) points[i] /= scale
            return points
        }
    }
}
