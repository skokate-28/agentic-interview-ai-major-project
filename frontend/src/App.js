import React, { useMemo, useState } from "react";
import { fetchDashboard, fetchNextQuestion, startInterview, submitAnswer } from "./api";
import HomeScreen from "./components/HomeScreen";
import InterviewScreen from "./components/InterviewScreen";
import FinalReportScreen from "./components/FinalReportScreen";
import DashboardScreen from "./components/DashboardScreen";

function App() {
  const [view, setView] = useState("home");
  const [name, setName] = useState("");
  const [userId, setUserId] = useState("");
  const [mode, setMode] = useState("text");
  const [resumeFile, setResumeFile] = useState(null);

  const [phase, setPhase] = useState("technical");
  const [currentSkill, setCurrentSkill] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [evaluationData, setEvaluationData] = useState(null);

  const [finalReport, setFinalReport] = useState(null);
  const [dashboard, setDashboard] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const canSubmitAnswer = useMemo(() => {
    return Boolean(answer.trim()) && !evaluationData;
  }, [answer, evaluationData]);

  const handleFileChange = (e) => {
    setResumeFile(e.target.files[0]);
  };

  async function handleStartInterview() {
    if (!resumeFile) {
      alert("Please upload your resume");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await startInterview({
        user_id: userId,
        name,
        mode,
        resume: resumeFile,
      });

      setSessionId(data.session_id || null);
      setPhase(data.phase || "technical");
      setCurrentSkill(data.current_skill || "");
      setQuestion(data.question || "");
      setAnswer("");
      setEvaluationData(null);
      setFinalReport(null);
      setDashboard(null);
      setView("interview");
      setLoading(false);
    } catch (err) {
      setLoading(false);
      alert("Failed to start interview. Please try again.");
      const message = err?.response?.data?.detail || err?.message || "Failed to start interview. Please try again.";
      setError(String(message));
    }
  }

  async function handleSubmitAnswer() {
    if (!canSubmitAnswer || loading) {
      return;
    }

    if (!sessionId) {
      alert("Session not initialized. Please restart interview.");
      return;
    }

    const answerText = String(answer ?? "");
    console.log("Payload:", {
      session_id: sessionId,
      answer: answerText,
      type: typeof answerText,
    });

    setLoading(true);
    setError("");

    try {
      const data = await submitAnswer({
        session_id: sessionId,
        answer: String(answerText),
      });

      if (data.status === "in_progress" || data.status === "retry") {
        setPhase(data.phase || phase);
        setCurrentSkill(data.current_skill || "");
        setQuestion(data.question || "");
        setEvaluationData(null);
        setAnswer("");

        if (data.status === "retry" && data.message) {
          setError(String(data.message));
        }
        return;
      }

      if (data.status === "evaluated") {
        setEvaluationData({
          score: data.score,
          bkt_probability: data.bkt_probability,
          bkt_width: data.bkt_width,
          next_difficulty: data.next_difficulty,
          next_question_type: data.next_question_type,
          weakness: data.weakness || null,
        });
        setAnswer("");
        return;
      }

      if (data.end || data.status === "completed" || data.status === "complete") {
        console.log("Interview completed");
        const response = data.final_report && Object.keys(data.final_report).length > 0
          ? data.final_report
          : {
              technical: data.technical || {},
              global: data.global || {},
              hr: data.hr || {},
            };
        console.log("FINAL REPORT DATA:", response);
        setEvaluationData(null);
        setFinalReport(response);
        setView("final");
        setAnswer("");
        return;
      }

      if (!data.question) {
        console.error("Invalid response:", data);
      }

      setError("Unexpected response from server.");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function handleProceedNextQuestion() {
    if (!sessionId || loading) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const data = await fetchNextQuestion({ session_id: sessionId });

      if (data.status === "next_question") {
        setPhase(data.phase || phase);
        setCurrentSkill(data.current_skill || "");
        setQuestion(data.question || "");
        setEvaluationData(null);
        return;
      }

      if (data.end || data.status === "completed" || data.status === "complete") {
        console.log("Interview completed");
        const response = data.final_report && Object.keys(data.final_report).length > 0
          ? data.final_report
          : {
              technical: data.technical || {},
              global: data.global || {},
              hr: data.hr || {},
            };
        console.log("FINAL REPORT DATA:", response);
        setEvaluationData(null);
        setFinalReport(response);
        setView("final");
        return;
      }

      setError("Unexpected response from server.");
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function handleOpenDashboard() {
    setLoading(true);
    setError("");

    try {
      const data = await fetchDashboard(userId);
      setDashboard(data);
      setView("dashboard");
    } catch (err) {
      const message = err?.response?.data?.detail || err?.message || "Failed to load dashboard.";
      setError(String(message));
    } finally {
      setLoading(false);
    }
  }

  function handleBackHome() {
    setView("home");
    setPhase("technical");
    setCurrentSkill("");
    setQuestion("");
    setAnswer("");
    setEvaluationData(null);
    setSessionId(null);
    setFinalReport(null);
    setDashboard(null);
    setError("");
  }

  return (
    <main className="app-shell">
      {view === "home" ? (
        <HomeScreen
          name={name}
          userId={userId}
          mode={mode}
          resumeFile={resumeFile}
          onNameChange={setName}
          onUserIdChange={setUserId}
          onModeChange={setMode}
          onResumeChange={handleFileChange}
          onStart={handleStartInterview}
          loading={loading}
          error={error}
        />
      ) : null}

      {view === "interview" ? (
        <InterviewScreen
          mode={mode}
          phase={phase}
          currentSkill={currentSkill}
          question={question}
          answer={answer}
          evaluationData={evaluationData}
          onAnswerChange={setAnswer}
          onSubmit={handleSubmitAnswer}
          onProceedNextQuestion={handleProceedNextQuestion}
          loading={loading}
          error={error}
        />
      ) : null}

      {view === "final" ? (
        <FinalReportScreen report={finalReport} onOpenDashboard={handleOpenDashboard} />
      ) : null}

      {view === "dashboard" ? (
        <DashboardScreen dashboard={dashboard} onBackHome={handleBackHome} />
      ) : null}
    </main>
  );
}

export default App;
