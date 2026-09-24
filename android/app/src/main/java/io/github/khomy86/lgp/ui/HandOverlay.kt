package io.github.khomy86.lgp.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.unit.dp
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import io.github.khomy86.lgp.recognition.HandFrame
import io.github.khomy86.lgp.ui.theme.Landmark

/**
 * Draws the hand skeleton over a mirrored, centre-cropped (FILL_CENTER) camera preview.
 */
@Composable
fun HandOverlay(hand: HandFrame?, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        if (hand == null) return@Canvas

        val scale = maxOf(size.width / hand.imageWidth, size.height / hand.imageHeight)
        val dx = (size.width - hand.imageWidth * scale) / 2
        val dy = (size.height - hand.imageHeight * scale) / 2
        val points = hand.landmarks.map {
            Offset(dx + (1 - it.x()) * hand.imageWidth * scale, dy + it.y() * hand.imageHeight * scale)
        }

        val stroke = 3.dp.toPx()
        HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
            drawLine(
                color = Color.White.copy(alpha = 0.8f),
                start = points[connection.start()],
                end = points[connection.end()],
                strokeWidth = stroke,
                cap = StrokeCap.Round,
            )
        }
        points.forEach { drawCircle(color = Landmark, radius = stroke * 1.6f, center = it) }
    }
}
