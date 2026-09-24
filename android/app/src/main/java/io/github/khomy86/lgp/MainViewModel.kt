package io.github.khomy86.lgp

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import com.google.mediapipe.framework.image.MPImage
import io.github.khomy86.lgp.recognition.Gesture
import io.github.khomy86.lgp.recognition.GestureCatalog
import io.github.khomy86.lgp.recognition.GestureRecognizer
import io.github.khomy86.lgp.recognition.Recognition
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class UiState(
    val gestures: List<Gesture> = emptyList(),
    val recognition: Recognition? = null,
    val error: String? = null,
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow(UiState())
    val state: StateFlow<UiState> = _state.asStateFlow()

    private val recognizer: GestureRecognizer? = try {
        val catalog = GestureCatalog.load(application)
        _state.update { it.copy(gestures = catalog.gestures) }
        GestureRecognizer(
            context = application,
            catalog = catalog,
            onResult = { recognition -> _state.update { it.copy(recognition = recognition) } },
            onError = { message -> Log.e(TAG, message) },
        )
    } catch (e: Exception) {
        Log.e(TAG, "Could not load the models", e)
        _state.update { it.copy(error = e.message ?: e.toString()) }
        null
    }

    fun process(image: MPImage) {
        recognizer?.process(image)
    }

    override fun onCleared() {
        recognizer?.close()
    }

    private companion object {
        const val TAG = "MainViewModel"
    }
}
