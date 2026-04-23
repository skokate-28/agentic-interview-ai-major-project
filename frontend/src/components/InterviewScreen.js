import React, { useEffect, useRef, useState } from "react";

function InterviewScreen({
  mode,
  phase,
  currentSkill,
  question,
  answer,
  evaluationData,
  onAnswerChange,
  onSubmit,
  onProceedNextQuestion,
  loading,
  error,
}) {
  const recognitionRef = useRef(null);
  const [listening, setListening] = useState(false);
  const [speechError, setSpeechError] = useState("");

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (_err) {
          // Ignore stop errors during cleanup.
        }
      }
    };
  }, []);

  function startSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setSpeechError("Speech recognition is not supported in this browser.");
      return;
    }

    setSpeechError("");

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || "";
      if (transcript.trim()) {
        onAnswerChange(transcript.trim());
      }
      try {
        recognition.stop();
      } catch (_err) {
        // Ignore stop errors if recognition already ended.
      }
    };

    recognition.onerror = (event) => {
      setSpeechError(`Speech recognition error: ${event.error}`);
      setListening(false);
    };

    recognition.onend = () => {
      setListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
    setListening(true);
  }

  function stopSpeechRecognition() {
    if (!recognitionRef.current) {
      return;
    }

    try {
      recognitionRef.current.stop();
    } catch (_err) {
      // Ignore stop errors if recognition is not active.
    }
    setListening(false);
  }

  return (
    <div className="panel">
      <h2>Interview In Progress</h2>
      <p>Phase: {phase}</p>
      {currentSkill ? <p>Current Skill: {currentSkill}</p> : null}

      <div className="question-box">{question || "Waiting for next question..."}</div>

      {evaluationData ? (
        <div className="panel" style={{ marginTop: 12 }}>
          <h3>Evaluation</h3>
          <p>Score: {evaluationData.score ?? "-"}</p>
          <p>BKT Probability: {evaluationData.bkt_probability ?? "-"}</p>
          <p>BKT Width: {evaluationData.bkt_width ?? "-"}</p>
          <p>Next Difficulty: {evaluationData.next_difficulty ?? "-"}</p>
          <p>Next Question Type: {evaluationData.next_question_type ?? "-"}</p>
          {evaluationData.weakness ? (
            <>
              <p>Weakness:</p>
              <p>Hint: {evaluationData.weakness.question_hint ?? "-"}</p>
              <p>Difficulty: {evaluationData.weakness.difficulty ?? "-"}</p>
            </>
          ) : null}
          <button type="button" onClick={onProceedNextQuestion} disabled={loading}>
            {loading ? "Loading..." : "Proceed to Next Question"}
          </button>
        </div>
      ) : null}

      {!evaluationData ? (
        mode === "text" ? (
          <>
            <label>Your Answer</label>
            <textarea
              rows={5}
              value={answer}
              onChange={(e) => onAnswerChange(e.target.value)}
              placeholder="Type your answer"
              disabled={loading}
            />
          </>
        ) : (
          <>
            <label>Your Answer</label>
            <textarea
              rows={5}
              value={answer}
              onChange={(e) => onAnswerChange(e.target.value)}
              placeholder="Use mic or edit the transcript here"
              disabled={loading}
            />

            <button
              type="button"
              onClick={listening ? stopSpeechRecognition : startSpeechRecognition}
              disabled={loading}
            >
              {listening ? "Stop Mic" : "Start Mic"}
            </button>

            {speechError ? <div className="error">{speechError}</div> : null}
          </>
        )
      ) : null}

      {!evaluationData ? (
        <button onClick={onSubmit} disabled={loading || !answer.trim()}>
          {loading ? "Submitting..." : "Submit Answer"}
        </button>
      ) : null}

      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}

export default InterviewScreen;
