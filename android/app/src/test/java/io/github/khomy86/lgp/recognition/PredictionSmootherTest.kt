package io.github.khomy86.lgp.recognition

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PredictionSmootherTest {

    @Test
    fun averageIsNullWhenEmpty() {
        assertNull(PredictionSmoother(3).average())
    }

    @Test
    fun averagesOnlyTheLastWindow() {
        val smoother = PredictionSmoother(2)
        smoother.add(floatArrayOf(1f, 0f))
        smoother.add(floatArrayOf(0f, 1f))
        smoother.add(floatArrayOf(0f, 1f))
        assertEquals(2, smoother.size)
        assertArrayEquals(floatArrayOf(0f, 1f), smoother.average(), 1e-6f)
    }

    @Test
    fun clearForgetsHistory() {
        val smoother = PredictionSmoother(3)
        smoother.add(floatArrayOf(0.5f, 0.5f))
        smoother.clear()
        assertEquals(0, smoother.size)
        assertNull(smoother.average())
    }
}
