package io.github.khomy86.lgp.recognition

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the on-device pipeline (MediaPipe -> normalisation -> TFLite classifier) on one
 * dataset photo per gesture. The photos were part of the training data, so this checks
 * that the app's pipeline is wired correctly, not how accurate the model is.
 */
@RunWith(AndroidJUnit4::class)
class RecognitionPipelineTest {

    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context
    private lateinit var catalog: GestureCatalog
    private lateinit var landmarker: HandLandmarker
    private lateinit var classifier: GestureClassifier

    @Before
    fun setUp() {
        catalog = GestureCatalog.load(appContext)
        classifier = GestureClassifier(appContext, catalog.gestures.size)
        landmarker = HandLandmarker.createFromOptions(
            appContext,
            HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(GestureRecognizer.HAND_LANDMARKER_ASSET).build())
                .setRunningMode(RunningMode.IMAGE)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.3f)
                .setMinHandPresenceConfidence(0.3f)
                .build()
        )
    }

    @After
    fun tearDown() {
        landmarker.close()
        classifier.close()
    }

    @Test
    fun recognisesEachSampleImage() {
        for (gesture in catalog.gestures) {
            val image = withMargin(loadSample(gesture.id))
            val hand = landmarker.detect(BitmapImageBuilder(image).build()).landmarks().firstOrNull()
            assertNotNull("No hand found in ${gesture.id}.jpg", hand)

            val scores = classifier.classify(LandmarkNormalizer.normalize(hand!!, image.width, image.height))
            val predicted = catalog.gestures[scores.indices.maxBy { scores[it] }]
            assertEquals("Wrong gesture for ${gesture.id}.jpg", gesture.id, predicted.id)
        }
    }

    private fun loadSample(id: String): Bitmap =
        testContext.assets.open("samples/$id.jpg").use { BitmapFactory.decodeStream(it) }

    /** The dataset photos are tight crops; like the camera, MediaPipe needs some context around the hand. */
    private fun withMargin(source: Bitmap): Bitmap {
        val pad = (maxOf(source.width, source.height) * 0.3f).toInt()
        val result = Bitmap.createBitmap(source.width + 2 * pad, source.height + 2 * pad, Bitmap.Config.ARGB_8888)
        Canvas(result).apply {
            drawColor(Color.BLACK)
            drawBitmap(source, pad.toFloat(), pad.toFloat(), null)
        }
        return result
    }
}
