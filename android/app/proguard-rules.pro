# MediaPipe and LiteRT load parts of themselves through JNI and reflection.
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
-keep class org.tensorflow.lite.** { *; }

# MediaPipe logs through Flogger, which finds its caller by class name on the stack.
-keep class com.google.common.flogger.** { *; }

# Referenced by MediaPipe's profiling APIs but not shipped in the AAR.
-dontwarn com.google.mediapipe.proto.**
