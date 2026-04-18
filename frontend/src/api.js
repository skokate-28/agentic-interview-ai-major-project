import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 30000,
});

export async function startInterview(payload) {
  const formData = new FormData();
  formData.append("name", payload.name);
  formData.append("user_id", payload.user_id);
  formData.append("mode", payload.mode);
  formData.append("resume", payload.resume);

  const response = await api.post("/start_interview", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export async function submitAnswer(payload) {
  const response = await api.post("/answer", {
    session_id: payload.session_id,
    answer: String(payload.answer ?? ""),
  });
  return response.data;
}

export async function fetchNextQuestion(payload) {
  const response = await api.post("/next_question", {
    session_id: payload.session_id,
  });
  return response.data;
}

export async function fetchDashboard(userId) {
  const response = await api.get(`/dashboard/${encodeURIComponent(userId)}`);
  return response.data;
}
