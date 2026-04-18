import React from "react";

function FinalReportScreen({ report, onOpenDashboard }) {
  const technical = report?.technical || {};
  const globalScores = report?.global || {};
  const hr = report?.hr || {};
  const technicalSkills = technical?.skills || {};

  return (
    <div className="panel">
      <h2>Final Report</h2>

      <section>
        <h3>Technical</h3>
        <p>Overall Score: {technical.overall_score ?? "-"}</p>
        <ul>
          {Object.entries(technicalSkills).map(([skill, score]) => (
            <li key={skill}>
              {skill}: {score}
            </li>
          ))}
          {Object.keys(technicalSkills).length === 0 ? <li>No skill scores available.</li> : null}
        </ul>
      </section>

      <section>
        <h3>Global</h3>
        <ul>
          <li>Communication: {globalScores.communication ?? "-"}</li>
          <li>Confidence: {globalScores.confidence ?? "-"}</li>
        </ul>
      </section>

      <section>
        <h3>HR</h3>
        <p>Overall Score: {hr.overall_score ?? "-"}</p>
        <ul>
          <li>Leadership: {hr.leadership ?? "-"}</li>
          <li>Problem Solving: {hr.problem_solving ?? "-"}</li>
          <li>Adaptability: {hr.adaptability ?? "-"}</li>
          <li>Teamwork: {hr.teamwork ?? "-"}</li>
        </ul>
      </section>

      <button onClick={onOpenDashboard}>Open Dashboard</button>
    </div>
  );
}

export default FinalReportScreen;
