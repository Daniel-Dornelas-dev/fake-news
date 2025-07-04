import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ texto: text }),
      });

      const data = await response.json();
      setResult(data.resultado);
    } catch (error: unknown) {
      if (error instanceof Error) {
        setResult("Erro ao consultar a API: " + error.message);
      } else {
        setResult("Erro ao consultar a API.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Verificador de Fake News</h1>
      <textarea
        placeholder="Cole aqui o texto da notícia..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Verificando..." : "Verificar"}
      </button>
      {result && <div className="result">{result}</div>}
    </div>
  );
}

export default App;
