package io.github.khomy86.lgp.recognition

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.hypot

object LandmarkNormalizer {
    const val NUM_LANDMARKS = 21
    const val NUM_FEATURES = NUM_LANDMARKS * 3

    /**
     * Makes landmarks independent of where the hand is in the frame and how big it looks:
     * converts them to pixel proportions, moves the wrist to the origin and divides by the
     * largest wrist distance. Must match `normalize_landmarks` in ml/landmarks.py.
     */
    fun normalize(landmarks: List<NormalizedLandmark>, width: Int, height: Int): FloatArray {
        val wrist = landmarks[0]
        val features = FloatArray(NUM_FEATURES)
        var scale = 0f
        landmarks.forEachIndexed { i, lm ->
            val x = (lm.x() - wrist.x()) * width
            val y = (lm.y() - wrist.y()) * height
            features[i * 3] = x
            features[i * 3 + 1] = y
            features[i * 3 + 2] = (lm.z() - wrist.z()) * width
            scale = maxOf(scale, hypot(x, y))
        }
        scale = maxOf(scale, 1e-6f)
        for (i in features.indices) features[i] /= scale
        return features
    }
}
