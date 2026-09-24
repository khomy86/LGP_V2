package io.github.khomy86.lgp.recognition

/** Averages the classifier output over the last [window] frames. */
class PredictionSmoother(private val window: Int) {
    private val history = ArrayDeque<FloatArray>()

    val size: Int get() = history.size

    fun add(scores: FloatArray) {
        history.addLast(scores)
        if (history.size > window) history.removeFirst()
    }

    fun clear() = history.clear()

    fun average(): FloatArray? {
        if (history.isEmpty()) return null
        val sum = FloatArray(history.first().size)
        for (scores in history) scores.forEachIndexed { i, s -> sum[i] += s }
        return FloatArray(sum.size) { sum[it] / history.size }
    }
}
