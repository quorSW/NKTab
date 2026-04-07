const sendBtn = document.getElementById("sendBtn");
const promptEl = document.getElementById("prompt");
const resultEl = document.getElementById("result");

sendBtn.addEventListener("click", async () => {
  const prompt = promptEl.value.trim();

  if (!prompt) {
    resultEl.textContent = "Введите запрос.";
    return;
  }

  resultEl.textContent = "Думаю...";

  try {
    const res = await fetch("http://127.0.0.1:8000/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        prompt,
        max_new_tokens: 120,
        temperature: 0.9
      })
    });

    const data = await res.json();
    resultEl.textContent = data.response || "Нет ответа.";
  } catch (err) {
    resultEl.textContent = "Ошибка подключения к локальной модели.";
  }
});
