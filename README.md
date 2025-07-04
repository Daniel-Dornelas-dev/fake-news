# 📰 API de Detecção de Fake News (PT-BR)

Este projeto utiliza **BERTimbau** e **Logistic Regression** para classificar textos curtos em português do Brasil como **notícias verdadeiras ou falsas**, com foco especial em frases sensíveis e curtas que envolvem ética, saúde e diversidade.

---

## 🚀 Funcionalidades

- 🔎 Classificação de frases como **Fake** ou **True**
- 💬 Suporte a frases curtas e sensíveis (ex: "água faz mal à saúde")
- 📚 Histórico de classificações realizadas
- 📡 API REST com **FastAPI**

---

## 📁 Estrutura de Pastas

```
app/
├── main.py                  # API principal (FastAPI)
├── data/                    # Vetores TF-IDF (ignorado no Git)
├── data_bert/               # Embeddings BERT e modelos (ignorado no Git)
├── Fake.br-Corpus/          # Dataset original e pré-processado
├── services/                # Lógica da API: classificação, status, histórico
├── treino/                  # Scripts de pré-processamento, treinamento e predição
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

---

## ⚙️ Como rodar o projeto

### 1. Clone o repositório e crie o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute a API com Uvicorn

```bash
uvicorn app.main:app --reload
```

### 4. Acesse a interface de testes (Swagger UI)

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 Endpoints

### `POST /api/classificar-noticia`
Classifica um texto como **Fake** ou **True**.

**Body JSON:**
```json
{
  "texto": "água faz mal à saúde"
}
```

**Resposta:**
```json
{
  "classificacao": "Fake",
  "confianca": 97.32,
  "data": "2025-06-28T12:00:00"
}
```

---

### `GET /api/historico`
Retorna o histórico de classificações realizadas.

### `GET /api/status`
Retorna status atual do modelo carregado (tipo, embeddings, versão).

---

## 🧠 Modelo

- **Embeddings:** `BERTimbau` (`neuralmind/bert-base-portuguese-cased`)
- **Classificador:** `LogisticRegression`
- **Treinamento:** `bert_train.py`
- **Dados:** Corpus Fake.Br + frases sociais e éticas criadas para melhorar a generalização

---

## 📄 Licença

Este projeto é de uso educacional e acadêmico. Para uso em produção, é importante aplicar filtros adicionais, controle de viés e validação contínua.

---

## 🙋‍♂️ Contato

Dúvidas ou sugestões? Contribuições são bem-vindas!