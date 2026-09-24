package io.github.khomy86.lgp.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.google.mediapipe.framework.image.MPImage
import io.github.khomy86.lgp.R
import io.github.khomy86.lgp.UiState
import io.github.khomy86.lgp.recognition.Gesture
import kotlin.math.roundToInt

@Composable
fun RecognitionScreen(state: UiState, onFrame: (MPImage) -> Unit) {
    val recognition = state.recognition

    Box(modifier = Modifier.fillMaxSize()) {
        CameraPreview(onFrame = onFrame, modifier = Modifier.fillMaxSize())
        HandOverlay(hand = recognition?.hand, modifier = Modifier.fillMaxSize())

        TopBar(
            latencyMs = recognition?.latencyMs,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .statusBarsPadding()
                .padding(16.dp),
        )

        ResultPanel(
            state = state,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
                .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.94f))
                .navigationBarsPadding()
                .padding(20.dp),
        )
    }
}

@Composable
private fun TopBar(latencyMs: Long?, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Pill(stringResource(R.string.app_name), fontWeight = FontWeight.SemiBold)
        if (latencyMs != null) Pill(stringResource(R.string.latency, latencyMs))
    }
}

@Composable
private fun Pill(text: String, fontWeight: FontWeight = FontWeight.Normal) {
    Text(
        text = text,
        color = Color.White,
        fontSize = 13.sp,
        fontWeight = fontWeight,
        modifier = Modifier
            .clip(RoundedCornerShape(50))
            .background(Color.Black.copy(alpha = 0.5f))
            .padding(horizontal = 12.dp, vertical = 6.dp),
    )
}

@Composable
private fun ResultPanel(state: UiState, modifier: Modifier = Modifier) {
    val recognition = state.recognition
    val gesture = recognition?.gesture
    val confidence by animateFloatAsState(recognition?.confidence ?: 0f, label = "confidence")

    Column(modifier = modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            text = stringResource(R.string.recognised_sign).uppercase(),
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = gesture?.name ?: "—",
            fontSize = 40.sp,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
        )
        Text(
            text = when {
                gesture != null -> gesture.meaning
                recognition?.hand != null || recognition?.scores != null -> stringResource(R.string.hold_steady)
                else -> stringResource(R.string.no_hand)
            },
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(8.dp))
        LinearProgressIndicator(
            progress = { confidence },
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp),
            strokeCap = StrokeCap.Round,
            trackColor = MaterialTheme.colorScheme.surfaceVariant,
            gapSize = 0.dp,
            drawStopIndicator = {},
        )
        Text(
            text = if (recognition?.scores != null) stringResource(R.string.confidence, (confidence * 100).roundToInt()) else " ",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(8.dp))
        GestureChips(state.gestures, active = gesture)
    }
}

@Composable
private fun GestureChips(gestures: List<Gesture>, active: Gesture?) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        gestures.forEach { gesture ->
            val selected = gesture == active
            val background by animateColorAsState(
                if (selected) MaterialTheme.colorScheme.primary else Color.Transparent,
                label = "chip",
            )
            Text(
                text = gesture.name,
                fontSize = 14.sp,
                color = if (selected) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier
                    .clip(RoundedCornerShape(50))
                    .background(background)
                    .border(1.dp, if (selected) Color.Transparent else MaterialTheme.colorScheme.outline, RoundedCornerShape(50))
                    .padding(horizontal = 12.dp, vertical = 6.dp),
            )
        }
    }
}
