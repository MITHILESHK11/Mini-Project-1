// src/components/RecognitionButton.tsx
"use client";

export default function RecognitionButton() {
  const handleStart = () => {
    alert("Run: `python core/realtime_recognition.py` in your terminal to start local camera recognition.");
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <button
        onClick={handleStart}
        className="w-full bg-purple-600 text-white py-3 px-6 rounded-md hover:bg-purple-700 transition-colors font-medium"
      >
        🎥 Start Recognition (Local Camera)
      </button>
      <p className="text-sm text-gray-500 mt-2 text-center">
        This will trigger your Python recognition script
      </p>
    </div>
  );
}
