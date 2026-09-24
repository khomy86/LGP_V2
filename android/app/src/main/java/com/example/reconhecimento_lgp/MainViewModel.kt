package com.example.reconhecimento_lgp

import androidx.lifecycle.ViewModel
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class MainViewModel : ViewModel() {
    private val _recognition = MutableStateFlow<GestureRecognizerHelper.Recognition?>(null)
    val recognition = _recognition.asStateFlow()

    private val _handLandmarkerResult = MutableStateFlow<HandLandmarkerResult?>(null)
    val handLandmarkerResult = _handLandmarkerResult.asStateFlow()

    fun onGestureRecognized(recognition: GestureRecognizerHelper.Recognition?) {
        _recognition.value = recognition
    }

    fun onHandLandmarkerResult(result: HandLandmarkerResult) {
        _handLandmarkerResult.value = result
    }
}
