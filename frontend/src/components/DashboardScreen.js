import React from "react";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

function DashboardScreen({ dashboard, onBackHome }) {
  const overallData = dashboard?.overall_score_history || [];
  const skillHistories = dashboard?.skill_histories || {};

  return (
    <div className="panel">
      <h2>User Dashboard</h2>
      <p>User: {dashboard?.name || dashboard?.user_id}</p>

      <section>
        <h3>Overall Score History</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={overallData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="session_id" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="score" stroke="#1f77b4" name="Overall" />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section>
        <h3>Per-Skill Histories</h3>
        {Object.keys(skillHistories).length === 0 ? (
          <p>No skill history available.</p>
        ) : (
          Object.entries(skillHistories).map(([skill, points]) => (
            <div key={skill} className="chart-block">
              <h4>{skill}</h4>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={points}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="session_id" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="score" stroke="#2ca02c" name={skill} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))
        )}
      </section>

      <button onClick={onBackHome}>Back to Home</button>
    </div>
  );
}

export default DashboardScreen;
