package io.github.khomy86.lgp.recognition

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Runs the hand-shape classifier exported by ml/export.py. Not thread-safe. */
class GestureClassifier(context: Context, numClasses: Int) : Closeable {
    private val interpreter = Interpreter(loadModel(context))
    private val input = ByteBuffer.allocateDirect(LandmarkNormalizer.NUM_FEATURES * Float.SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
    private val output = Array(1) { FloatArray(numClasses) }

    fun classify(features: FloatArray): FloatArray {
        input.rewind()
        features.forEach { input.putFloat(it) }
        input.rewind()
        interpreter.run(input, output)
        return output[0].copyOf()
    }

    override fun close() = interpreter.close()

    private companion object {
        const val MODEL_ASSET = "gesture_classifier.tflite"

        fun loadModel(context: Context): MappedByteBuffer =
            context.assets.openFd(MODEL_ASSET).use { fd ->
                FileInputStream(fd.fileDescriptor).use { stream ->
                    stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
    }
}
