package io.github.khomy86.lgp.recognition

import android.content.Context
import android.os.SystemClock
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.io.Closeable

/** A detected hand, in the coordinates of the (upright) camera frame it was found in. */
data class HandFrame(
    val landmarks: List<NormalizedLandmark>,
    val imageWidth: Int,
    val imageHeight: Int,
)

data class Recognition(
    val hand: HandFrame?,
    /** Smoothed probability per gesture, or null while there isn't enough history. */
    val scores: List<Float>?,
    /** The recognised gesture, or null when nothing reaches the confidence threshold. */
    val gesture: Gesture?,
    val confidence: Float,
    val latencyMs: Long,
)

/**
 * Camera frame -> MediaPipe hand landmarks -> normalised features -> classifier -> smoothing.
 *
 * Frames are processed asynchronously; results arrive on a MediaPipe thread.
 */
class GestureRecognizer(
    context: Context,
    val catalog: GestureCatalog,
    private val onResult: (Recognition) -> Unit,
    private val onError: (String) -> Unit,
) : Closeable {

    private val classifier = GestureClassifier(context, catalog.gestures.size)
    private val smoother = PredictionSmoother(catalog.smoothingWindow)
    private var framesWithoutHand = 0

    private val handLandmarker = HandLandmarker.createFromOptions(
        context,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(HAND_LANDMARKER_ASSET).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .setResultListener(this::handleResult)
            .setErrorListener { e -> onError(e.message ?: "Hand landmarker error") }
            .build()
    )

    fun process(image: MPImage) {
        handLandmarker.detectAsync(image, SystemClock.uptimeMillis())
    }

    private fun handleResult(result: HandLandmarkerResult, image: MPImage) {
        val latency = SystemClock.uptimeMillis() - result.timestampMs()
        val landmarks = result.landmarks().firstOrNull()

        if (landmarks == null) {
            // Short detection dropouts are common mid-sign; keep the history through them.
            if (++framesWithoutHand > MAX_FRAMES_WITHOUT_HAND) smoother.clear()
        } else {
            framesWithoutHand = 0
            val features = LandmarkNormalizer.normalize(landmarks, image.width, image.height)
            smoother.add(classifier.classify(features))
        }

        val scores = if (smoother.size >= MIN_FRAMES) smoother.average()?.toList() else null
        val best = scores?.indices?.maxBy { scores[it] }
        val confidence = if (best != null) scores[best] else 0f
        val gesture = best?.takeIf { confidence >= catalog.minConfidence }?.let { catalog.gestures[it] }

        onResult(
            Recognition(
                hand = landmarks?.let { HandFrame(it, image.width, image.height) },
                scores = scores,
                gesture = gesture,
                confidence = confidence,
                latencyMs = latency,
            )
        )
    }

    override fun close() {
        handLandmarker.close()
        classifier.close()
    }

    companion object {
        const val HAND_LANDMARKER_ASSET = "hand_landmarker.task"
        private const val MIN_FRAMES = 5
        private const val MAX_FRAMES_WITHOUT_HAND = 5
    }
}
