package com.example.reconhecimento_lgp

import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors
import kotlin.math.roundToInt

@Composable
fun MainScreen(viewModel: MainViewModel, gestureRecognizerHelper: GestureRecognizerHelper) {
    val recognition by viewModel.recognition.collectAsStateWithLifecycle()
    val handLandmarkerResult by viewModel.handLandmarkerResult.collectAsStateWithLifecycle()
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        onDispose { cameraExecutor.shutdown() }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { context ->
                val previewView = PreviewView(context)
                val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                    val imageAnalyzer = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor) { imageProxy ->
                                try {
                                    gestureRecognizerHelper.recognizeLiveStream(imageProxy.toMPImage())
                                } catch (e: Exception) {
                                    Log.e("MainScreen", "Error processing image", e)
                                } finally {
                                    imageProxy.close()
                                }
                            }
                        }
                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_FRONT_CAMERA,
                            preview,
                            imageAnalyzer
                        )
                    } catch (e: Exception) {
                        Log.e("MainScreen", "Camera binding failed", e)
                    }
                }, ContextCompat.getMainExecutor(context))
                previewView
            },
            modifier = Modifier.fillMaxSize()
        )
        Overlay(handLandmarkerResult, noHandText = stringResource(R.string.no_hand_detected))
        RecognitionLabel(
            recognition = recognition,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 48.dp)
        )
    }
}

private val displayNames = mapOf(
    "agua" to "Água",
    "bom dia" to "Bom dia",
    "nao" to "Não",
    "ola" to "Olá",
    "por favor" to "Por favor",
    "sim" to "Sim",
)

@Composable
fun RecognitionLabel(recognition: GestureRecognizerHelper.Recognition?, modifier: Modifier = Modifier) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier
            .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(16.dp))
            .padding(horizontal = 24.dp, vertical = 12.dp)
    ) {
        if (recognition == null) {
            Text(text = "…", fontSize = 32.sp, color = Color.White)
        } else {
            Text(
                text = displayNames[recognition.label] ?: recognition.label,
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )
            Text(
                text = "${(recognition.confidence * 100).roundToInt()}%",
                fontSize = 14.sp,
                color = Color.White.copy(alpha = 0.7f)
            )
        }
    }
}

@Composable
fun Overlay(result: HandLandmarkerResult?, noHandText: String) {
    val textPaint = remember {
        Paint().apply {
            color = android.graphics.Color.WHITE
            textSize = 50f
            textAlign = Paint.Align.CENTER
        }
    }

    Canvas(modifier = Modifier.fillMaxSize()) {
        val hands = result?.landmarks().orEmpty()
        if (hands.isEmpty()) {
            drawContext.canvas.nativeCanvas.drawText(noHandText, size.width / 2, size.height / 2, textPaint)
            return@Canvas
        }

        // A pré-visualização da câmara frontal aparece espelhada, a análise não.
        fun toScreen(x: Float, y: Float) = Offset((1f - x) * size.width, y * size.height)

        hands.forEach { landmarks ->
            HandLandmarker.HAND_CONNECTIONS.forEach { connection ->
                val start = landmarks[connection.start()]
                val end = landmarks[connection.end()]
                drawLine(
                    color = Color.Red,
                    start = toScreen(start.x(), start.y()),
                    end = toScreen(end.x(), end.y()),
                    strokeWidth = 3f
                )
            }

            val padding = 0.05f
            val topLeft = toScreen(
                (landmarks.maxOf { it.x() } + padding).coerceAtMost(1f),
                (landmarks.minOf { it.y() } - padding).coerceAtLeast(0f)
            )
            val bottomRight = toScreen(
                (landmarks.minOf { it.x() } - padding).coerceAtLeast(0f),
                (landmarks.maxOf { it.y() } + padding).coerceAtMost(1f)
            )
            drawRect(
                color = Color.Green,
                topLeft = topLeft,
                size = Size(bottomRight.x - topLeft.x, bottomRight.y - topLeft.y),
                style = Stroke(width = 8f)
            )
        }
    }
}

private fun ImageProxy.toMPImage(): MPImage {
    var bitmap = toBitmap()
    val rotation = imageInfo.rotationDegrees
    if (rotation != 0) {
        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
    return BitmapImageBuilder(bitmap).build()
}
