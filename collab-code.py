# =============================================================
# 🔹 SEO Query Fan-Out & Cosine Similarity Tool (Colab Version)
# =============================================================

# ==============================================
# 🧩 GEREKLİ KÜTÜPHANELERİ YÜKLE
# ==============================================
!pip install -q sentence-transformers openai numpy scikit-learn pandas

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import re, time

# ==============================================
# 🔐 HUGGINGFACE TOKEN & OPENROUTER API ANAHTARLARI
# ==============================================
HF_TOKEN = "hf_xxxxxxxx"  # 👈 HuggingFace token'ını buraya yaz
OPENROUTER_KEYS = {
    "llama4":  "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "deepseek":"sk-or-v1-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
    "qwen":    "sk-or-v1-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
}

# ==============================================
# 🧠 EMBEDDINGGEMMA MODELİNİ YÜKLE
# ==============================================
embedding_model = SentenceTransformer(
    "google/embeddinggemma-300m",
    cache_folder="/content/cache",
    use_auth_token=HF_TOKEN
)
print("✅ EmbeddingGemma başarıyla yüklendi!")

# ==============================================
# ⚙️ OPENROUTER MODELLERİ
# ==============================================
MODELS = {
    "llama4":  "meta-llama/llama-4-maverick:free",
    "deepseek":"deepseek/deepseek-chat-v3.1:free",
    "qwen":    "qwen/qwen3-14b:free"
}

clients = {
    name: OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    for name, key in OPENROUTER_KEYS.items()
}
print("✅ OpenRouter modelleri yüklendi:", list(clients.keys()))

# ==============================================
# 💬 QUERY FAN-OUT OLUŞTURMA
# ==============================================
def generate_fanout_queries(seed_query, n=15, max_retries=3):
    """
    Anahtar kelimeden 15 farklı Google tarzı sorgu üretir.
    """
    prompt = f"""
    Sen deneyimli bir SEO uzmanısın.
    Aşağıdaki anahtar kelimeden {n} farklı, doğal Google arama sorgusu üret.
    Dil otomatik olarak algılansın ve aynı dilde üretim yap.

    Anahtar kelime:
    "{seed_query}"

    🔹 Kurallar:
    - Bilgilendirici, yönlendirici, ticari ve işlemsel sorgular dengeli olmalı.
    - Kopya veya benzer sorgular üretme.
    - Yalnızca numaralı liste olarak çıktı ver.
    """

    for model_name in ["llama4", "deepseek", "qwen"]:
        for attempt in range(max_retries):
            try:
                print(f"⚙️ Kullanılan model: {model_name} (deneme {attempt+1})")
                client = clients[model_name]
                response = client.chat.completions.create(
                    model=MODELS[model_name],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                    max_tokens=400
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"❌ {model_name} başarısız: {e}")
                if attempt < max_retries - 1:
                    print("🔁 Tekrar deneniyor...")
                    time.sleep(5)
                else:
                    print("⏳ Sonraki modele geçiliyor...")
                    time.sleep(2)
                    break
    raise RuntimeError("🚨 Tüm modeller başarısız oldu. Token veya limitleri kontrol edin.")

# ==============================================
# 📈 COSINE SIMILARITY HESAPLAMA
# ==============================================
def analyze_similarity(seed_query, generated_text):
    """
    Anahtar kelime ile üretilen sorgular arasındaki semantik benzerliği hesaplar.
    """
    lines = [re.sub(r"^\d+\.\s*", "", l).strip() for l in generated_text.split("\n") if len(l.strip()) > 3]
    queries = [seed_query] + lines

    embeddings = embedding_model.encode(queries)
    base_emb = embeddings[0].reshape(1, -1)
    scores = cosine_similarity(base_emb, embeddings[1:])[0]

    results = sorted(zip(queries[1:], scores), key=lambda x: -x[1])
    print("\n🔎 Cosine Similarity Sonuçları:")
    for q, s in results:
        print(f"{q} → {s:.3f}")

    return pd.DataFrame(results, columns=["Sorgu", "Cosine_Benzerlik"])

# ==============================================
# 🚀 ÇALIŞTIRMA BLOĞU
# ==============================================
RUN_MULTI = False   # 👈 True = birden fazla sorgu / False = tek sorgu
all_results = []

if not RUN_MULTI:
    seed_query = "seo danışmanlığı"   # 👈 Buraya sorgunu yaz
    print(f"\n🔹 Anahtar Kelime: {seed_query}")
    generated = generate_fanout_queries(seed_query, n=15)
    df = analyze_similarity(seed_query, generated)
    df["Anahtar_Kelime"] = seed_query
    all_results.append(df)
else:
    seed_queries = ["seo hizmeti", "e-ticaret seo", "yerel seo danışmanlığı"]
    for query in seed_queries:
        print("\n==============================")
        print(f"🔹 Anahtar Kelime: {query}")
        generated = generate_fanout_queries(query, n=15)
        df = analyze_similarity(query, generated)
        df["Anahtar_Kelime"] = query
        all_results.append(df)
        time.sleep(2)

# ==============================================
# 💾 SONUÇLARI KAYDET
# ==============================================
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("query_fanout_results.csv", index=False)
print("\n📁 Sonuçlar kaydedildi: query_fanout_results.csv")
