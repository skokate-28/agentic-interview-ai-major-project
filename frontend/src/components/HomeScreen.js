import React from "react";

function HomeScreen({
  name,
  userId,
  mode,
  resumeFile,
  onNameChange,
  onUserIdChange,
  onModeChange,
  onResumeChange,
  onStart,
  loading,
  error,
}) {
  return (
    <div className="panel">
      <h1>Adaptive AI Interview</h1>
      <p>Start your interview in text mode or audio mode (UI only).</p>

      <label>Name</label>
      <input value={name} onChange={(e) => onNameChange(e.target.value)} placeholder="Enter your name" />

      <label>User ID</label>
      <input value={userId} onChange={(e) => onUserIdChange(e.target.value)} placeholder="Enter unique user id" />

      <label>Mode</label>
      <select value={mode} onChange={(e) => onModeChange(e.target.value)}>
        <option value="text">Text</option>
        <option value="audio">Audio (UI only)</option>
      </select>

      <label>Upload Resume</label>
      <input type="file" accept=".pdf,.txt" onChange={onResumeChange} />

      <button onClick={onStart} disabled={loading || !name.trim() || !userId.trim() || !resumeFile}>
        {loading ? "Starting..." : "Start Interview"}
      </button>

      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}

export default HomeScreen;
