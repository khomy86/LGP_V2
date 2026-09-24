package com.example.reconhecimento_lgp

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class NormalizeLandmarksTest {

    private val raw = floatArrayOf(
        0.5751f, 0.7383f, -0.0698f,
        0.3351f, 0.3801f, -0.0119f,
        0.2032f, 0.6927f, -0.0521f,
        0.4808f, 0.3818f, -0.0195f,
        0.3529f, 0.4670f, -0.0807f,
        0.5321f, 0.7973f, 0.0936f,
        0.5733f, 0.7934f, -0.0570f,
        0.2961f, 0.5675f, 0.0344f,
        0.2214f, 0.5089f, -0.0399f,
        0.7503f, 0.5775f, 0.0748f,
        0.4981f, 0.3485f, 0.0324f,
        0.3154f, 0.6152f, -0.0737f,
        0.4217f, 0.2022f, 0.0690f,
        0.2927f, 0.3606f, 0.0890f,
        0.5059f, 0.7083f, 0.0808f,
        0.6451f, 0.2549f, 0.0139f,
        0.5047f, 0.7228f, -0.0709f,
        0.5589f, 0.2356f, -0.0615f,
        0.3938f, 0.2901f, 0.0856f,
        0.4277f, 0.7872f, 0.0105f,
        0.5630f, 0.5828f, -0.0639f,
    )

    // Gerado com ml/landmarks.py: normalize_landmarks(raw, 480, 640)
    private val expected = floatArrayOf(
        0.000000f, 0.000000f, 0.000000f, -0.328284f, -0.653285f, 0.079198f, -0.508703f, -0.083165f, 0.024211f,
        -0.128988f, -0.650184f, 0.068803f, -0.303936f, -0.494797f, -0.014910f, -0.058817f, 0.107604f, 0.223506f,
        -0.002462f, 0.100491f, 0.017508f, -0.381630f, -0.311505f, 0.142530f, -0.483808f, -0.418379f, 0.040899f,
        0.239647f, -0.293267f, 0.197791f, -0.105324f, -0.710917f, 0.139794f, -0.355230f, -0.224510f, -0.005335f,
        -0.209828f, -0.977738f, 0.189857f, -0.386280f, -0.688849f, 0.217214f, -0.094655f, -0.054714f, 0.205998f,
        0.095749f, -0.881624f, 0.114489f, -0.096297f, -0.028269f, -0.001505f, -0.022159f, -0.916823f, 0.011353f,
        -0.247991f, -0.817426f, 0.212564f, -0.201621f, 0.089184f, 0.109838f, -0.016551f, -0.283601f, 0.008070f,
    )

    private val hand = raw.toList().chunked(3).map { (x, y, z) -> NormalizedLandmark.create(x, y, z) }

    @Test
    fun matchesPythonImplementation() {
        assertArrayEquals(expected, GestureRecognizerHelper.normalizeLandmarks(hand, 480, 640), 1e-5f)
    }

    @Test
    fun isInvariantToHandPositionAndSize() {
        val moved = hand.map { NormalizedLandmark.create(it.x() * 0.5f + 0.1f, it.y() * 0.5f + 0.2f, it.z() * 0.5f) }
        assertArrayEquals(
            GestureRecognizerHelper.normalizeLandmarks(hand, 480, 640),
            GestureRecognizerHelper.normalizeLandmarks(moved, 480, 640),
            1e-5f
        )
    }

    @Test
    fun farthestPointFromWristHasUnitDistance() {
        val points = GestureRecognizerHelper.normalizeLandmarks(hand, 480, 640)
        val maxDistance = (0 until 21).maxOf { i -> kotlin.math.hypot(points[i * 3], points[i * 3 + 1]) }
        assertEquals(1f, maxDistance, 1e-5f)
    }
}
