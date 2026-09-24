package io.github.khomy86.lgp.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

val Accent = Color(0xFF3987E5)
val Landmark = Color(0xFF6DA7EC)

// A camera app reads best on a dark UI, whatever the system theme.
private val colors = darkColorScheme(
    primary = Accent,
    onPrimary = Color.White,
    background = Color(0xFF111110),
    surface = Color(0xFF1A1A19),
    surfaceVariant = Color(0xFF242422),
    onSurface = Color.White,
    onSurfaceVariant = Color(0xFFC3C2B7),
    outline = Color(0xFF3A3A36),
)

@Composable
fun LgpTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = colors, content = content)
}
